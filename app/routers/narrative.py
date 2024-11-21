#routers/narrative.py
from fastapi import APIRouter, Depends, Query, HTTPException, Request, Header, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from app.utils.database import get_db
from app.models.models import WayneEnterprise, Article, Organization, MetricDefinition
import app.models.models as models
from sqlalchemy import func, text
from openai import OpenAI
import random
import json
import os
from decimal import Decimal
from app.connectors.connector_factory import ConnectorFactory
from dotenv import load_dotenv
import logging
import re
from app.utils.config import settings
from fastapi.responses import StreamingResponse
from io import BytesIO
import pdfkit
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pydantic import BaseModel, Field
from uuid import uuid4
import matplotlib.pyplot as plt
import base64
from reportlab.lib.utils import ImageReader
from io import BytesIO
import io
from reportlab.platypus import Image
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from io import BytesIO
import base64
from reportlab.lib.enums import TA_CENTER 
from app.utils.auth import get_current_user
from app.routers.data_source import get_all_connections_info, fetch_data_from_all_sources
from app.schemas.schemas import (
    Visualization,
    GraphData,
    NewsArticle,
    NewsFeed,
    ArticleSourceInfo,
    SourceInfo,
    MetricSourceInfo
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

router = APIRouter()

# Set up OpenAI client with GPT-3.5 Turbo
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PDFRequest(BaseModel):
    article_id: str
    chart_image: str


def analyze_metric_characteristics(metric_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze metric characteristics to determine visualization properties."""
    try:
        current = float(data.get('current', 0))
        previous = float(data.get('previous', 0))
        change = float(data.get('change', 0))
        change_percentage = float(data.get('change_percentage', 0))

        # Analyze value ranges and patterns
        characteristics = {
            'metric_name': metric_name,  # Add metric name to characteristics
            'is_percentage': 0 <= current <= 100 and 0 <= previous <= 100,
            'is_ratio': 0 <= current <= 1 and 0 <= previous <= 1,
            'is_large_number': current > 1000 or previous > 1000,
            'has_significant_change': abs(change_percentage) > 10,
            'has_minimal_change': abs(change_percentage) < 1,
            'is_zero_or_constant': current == 0 and previous == 0,
            'value_range': max(abs(current), abs(previous)),
            'decimal_places': len(str(current).split('.')[-1]) if '.' in str(current) else 0
        }

        # Add pattern-based characteristics
        name_patterns = detect_metric_patterns(metric_name)
        characteristics.update(name_patterns)

        return characteristics

    except (TypeError, ValueError) as e:
        logger.error(f"Error analyzing metric {metric_name}: {str(e)}")
        return {'metric_name': metric_name}  # Return at least the metric name

def detect_metric_patterns(metric_name: str) -> Dict[str, bool]:
    """Detect patterns in metric name to understand its nature."""
    name_lower = metric_name.lower()
    words = set(name_lower.split('_'))

    # Define pattern categories dynamically
    pattern_categories = {
        'time_based': {'time', 'duration', 'period', 'frequency', 'interval'},
        'financial': {'revenue', 'cost', 'price', 'spend', 'budget', 'sales', 'monetary'},
        'percentage_based': {'percentage', 'ratio', 'rate', 'share', 'proportion'},
        'count_based': {'count', 'total', 'number', 'quantity', 'sum'},
        'satisfaction': {'satisfaction', 'rating', 'score', 'feedback', 'review'},
        'performance': {'performance', 'efficiency', 'productivity', 'output'},
        'growth': {'growth', 'increase', 'decrease', 'change', 'delta'},
        'customer': {'customer', 'client', 'user', 'subscriber', 'member'}
    }

    # Check which patterns match
    patterns = {
        f'is_{category}': bool(words & pattern_words)
        for category, pattern_words in pattern_categories.items()
    }

    return patterns

def determine_best_visualization(characteristics: Dict[str, Any]) -> Dict[str, str]:
    """Determine the best visualization type based on metric characteristics."""
    metric_name = characteristics.get('metric_name', '').lower()
    
    # Start with visualization type determination
    chart_type = 'line'  # default fallback
    
    # Financial metrics - Use bar charts for clear comparison
    if characteristics.get('is_financial') or any(term in metric_name for term in ['salary', 'revenue', 'cost', 'price', 'budget']):
        chart_type = 'bar'
    
    # Percentage/Ratio metrics - Use pie charts for parts of a whole
    elif characteristics.get('is_percentage_based') or any(term in metric_name for term in ['ratio', 'rate', 'share', 'distribution']):
        chart_type = 'pie' if characteristics.get('value_range', 0) <= 100 else 'donut'
    
    # Performance/Score metrics - Use gauge charts for ratings
    elif any(term in metric_name for term in ['satisfaction', 'rating', 'score', 'performance']):
        if characteristics.get('value_range', 0) <= 5:  # Typical rating scale
            chart_type = 'gauge'
        else:
            chart_type = 'radar'
    
    # Time-based metrics - Use area charts for trends
    elif characteristics.get('is_time_based') or any(term in metric_name for term in ['duration', 'period', 'time']):
        chart_type = 'area'
    
    # Count-based metrics - Use column charts for comparison
    elif characteristics.get('is_count_based') or characteristics.get('is_large_number'):
        chart_type = 'column'
    
    # Growth/Change metrics - Use line charts for progression
    elif any(term in metric_name for term in ['growth', 'trend', 'progress', 'change']):
        chart_type = 'line'
    
    # Significant changes - Use bar charts to emphasize difference
    elif characteristics.get('has_significant_change'):
        chart_type = 'bar'
    
    # Multiple categories/comparison - Use radar charts
    elif characteristics.get('has_multiple_categories'):
        chart_type = 'radar'
    
    # Special cases based on value patterns
    if characteristics.get('is_zero_or_constant'):
        chart_type = 'bar'  # Better for showing stable values
    elif characteristics.get('has_minimal_change'):
        chart_type = 'area'  # Good for subtle changes

    # Determine stack type
    stack_type = None
    if characteristics.get('is_percentage_based') and chart_type in ['bar', 'column', 'area']:
        stack_type = 'normal'

    # Determine if points should be shown
    show_points = chart_type in ['line', 'area']

    # Determine if labels should be shown
    show_labels = chart_type in ['bar', 'column', 'pie', 'donut', 'gauge']

    # Create visualization configuration
    visualization = {
        'type': chart_type,
        'stack_type': stack_type,
        'show_points': show_points,
        'show_labels': show_labels,
        'axis_label': determine_axis_label(characteristics),
        'value_format': determine_value_format(characteristics)
    }

    return visualization

def determine_axis_label(characteristics: Dict[str, Any]) -> str:
    """Determine appropriate axis label based on metric characteristics."""
    if characteristics.get('is_financial'):
        return 'Amount ($)'
    elif characteristics.get('is_percentage_based'):
        return 'Percentage (%)'
    elif characteristics.get('is_count_based'):
        return 'Count'
    elif characteristics.get('is_satisfaction'):
        return 'Rating'
    elif characteristics.get('is_time_based'):
        return 'Duration'
    elif characteristics.get('is_performance'):
        return 'Performance'
    else:
        return 'Value'

def determine_value_format(characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Determine value formatting based on metric characteristics."""
    format_config = {
        'type': 'number',
        'decimal_places': 0,
        'prefix': '',
        'suffix': '',
        'use_grouping': characteristics.get('is_large_number', False)
    }

    if characteristics.get('is_financial'):
        format_config.update({
            'type': 'currency',
            'prefix': '$',
            'decimal_places': 2
        })
    elif characteristics.get('is_percentage_based'):
        format_config.update({
            'type': 'percentage',
            'suffix': '%',
            'decimal_places': 1
        })
    elif characteristics.get('is_satisfaction'):
        format_config.update({
            'type': 'decimal',
            'decimal_places': characteristics.get('decimal_places', 1)
        })

    return format_config

def enhance_graph_data(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance graph data with visualization properties."""
    enhanced_data = {}

    for metric_name, metric_data in graph_data.items():
        # Analyze metric characteristics
        characteristics = analyze_metric_characteristics(metric_name, metric_data)
        
        # Determine visualization properties
        visualization = determine_best_visualization(characteristics)
        
        # Enhance metric data
        enhanced_data[metric_name] = {
            **metric_data,
            'visualization': visualization,
            'characteristics': characteristics
        }

    return enhanced_data

def calculate_graph_data(current_data: Dict[str, Any], previous_data: Dict[str, Any]) -> Dict[str, GraphData]:
    """Calculate graph data dynamically for all numeric metrics."""
    graph_data = {}
    
    # Process all metrics from current data
    for metric_name, current_value in current_data.items():
        try:
            # Skip non-numeric values and special fields
            if metric_name in ['top_product', 'top_location'] or not isinstance(current_value, (int, float, Decimal)):
                continue
                
            # Convert values to float for consistency
            current = float(current_value)
            previous = float(previous_data.get(metric_name, 0))
            
            # Calculate changes
            change = current - previous
            change_percentage = (change / previous * 100) if previous != 0 else 0
            
            # Determine visualization properties
            characteristics = analyze_metric_characteristics(metric_name, {
                'current': current,
                'previous': previous,
                'change': change,
                'change_percentage': change_percentage
            })
            vis_props = determine_best_visualization(characteristics)
            
            # Create graph data entry with visualization
            graph_data[metric_name] = GraphData(
                current=current,
                previous=previous,
                change=change,
                change_percentage=change_percentage,
                visualization=Visualization(
                    type=vis_props['type'],
                    axis_label=vis_props['axis_label'],
                    value_format=vis_props['value_format'],
                    show_points=vis_props['show_points'],
                    stack_type=vis_props['stack_type'],
                    show_labels=vis_props['show_labels']
                )
            )
            
            logger.info(f"Calculated graph data for {metric_name}: current={current}, previous={previous}, change={change}, change_percentage={change_percentage}, visualization_type={vis_props['type']}")
            
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Error calculating graph data for {metric_name}: {str(e)}")
            graph_data[metric_name] = GraphData(
                current=0,
                previous=0,
                change=0,
                change_percentage=0,
                visualization=Visualization(
                    type='line',
                    axis_label='Value',
                    value_format={'type': 'number', 'decimal_places': 0, 'prefix': '', 'suffix': ''},
                    show_points=True,
                    stack_type=None,
                    show_labels=False
                )
            )
    
    return graph_data


def get_metric_categories(metrics_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Dynamically categorize metrics based on their names and values."""
    all_metrics = {k: v for k, v in metrics_data.items() if k != 'graph_data'}
    categories = {}
    
    # Define category patterns
    category_patterns = {
        "Performance": [
            r'performance', r'rating', r'score', r'efficiency', r'productivity',
            r'completed', r'achievement', r'kpi', r'success_rate'
        ],
        "Financial": [
            r'revenue', r'cost', r'salary', r'spend', r'budget', r'profit',
            r'margin', r'price', r'financial', r'monetary'
        ],
        "Customer": [
            r'customer', r'satisfaction', r'nps', r'retention', r'churn',
            r'feedback', r'complaint', r'service_level'
        ],
        "Employee": [
            r'employee', r'staff', r'training', r'turnover', r'resignation',
            r'satisfaction', r'engagement', r'attendance'
        ],
        "Operational": [
            r'hours', r'time', r'duration', r'workload', r'capacity',
            r'utilization', r'availability', r'uptime'
        ],
        "Growth": [
            r'growth', r'increase', r'trend', r'expansion', r'rate',
            r'acceleration', r'development'
        ],
        "Quality": [
            r'quality', r'defect', r'error', r'accuracy', r'precision',
            r'compliance', r'standard'
        ],
        "Product": [
            r'product', r'unit', r'sales', r'inventory', r'stock',
            r'feature', r'specification'
        ]
    }
    
    # Categorize each metric
    for metric_name, metric_value in all_metrics.items():
        metric_assigned = False
        
        # Try to match metric name with category patterns
        for category, patterns in category_patterns.items():
            if any(re.search(pattern, metric_name.lower()) for pattern in patterns):
                if category not in categories:
                    categories[category] = []
                categories[category].append(metric_name)
                metric_assigned = True
                break
        
        # If no category matched, add to "Other"
        if not metric_assigned:
            if "Other" not in categories:
                categories["Other"] = []
            categories["Other"].append(metric_name)
    
    return categories

def determine_chart_type(metric_name: str, metric_data: Dict[str, Any]) -> str:
    """Determine the most suitable chart type based on metric characteristics."""
    
    # Extract useful characteristics
    metric_lower = metric_name.lower()
    current_value = float(metric_data.get('current', 0))
    previous_value = float(metric_data.get('previous', 0))
    change = current_value - previous_value
    change_percentage = metric_data.get('change_percentage', 0)
    
    # Patterns for different chart types
    patterns = {
        'bar': [
            # Absolute values that benefit from direct comparison
            r'revenue', r'cost', r'salary', r'spend', r'budget',
            r'count', r'total', r'number', r'amount',
            # Metrics where magnitude matters
            r'sales', r'units', r'visitors', r'views'
        ],
        'line': [
            # Metrics showing trends or continuous change
            r'growth', r'trend', r'rate', r'progress',
            # Time-based metrics
            r'time', r'duration', r'period', r'frequency',
            # Performance indicators
            r'score', r'rating', r'index', r'level'
        ],
        'pie': [
            # Proportional or percentage-based metrics
            r'percentage', r'ratio', r'distribution', r'share',
            r'allocation', r'composition', r'breakdown',
            # Metrics naturally showing parts of a whole
            r'market_share', r'utilization'
        ],
        'scatter': [
            # Metrics showing correlation or distribution
            r'correlation', r'distribution', r'scatter',
            r'relationship', r'variance'
        ]
    }
    
    # Special cases based on value characteristics
    if abs(change_percentage) > 50:
        # Large changes are better shown with bar charts
        return 'bar'
    
    if any(current_value == previous_value for val in [0, 100]):
        # Fixed values or percentages work well with pie charts
        return 'pie'
    
    # Check metric name against patterns
    for chart_type, pattern_list in patterns.items():
        if any(re.search(pattern, metric_lower) for pattern in pattern_list):
            return chart_type
            
    # Additional rules based on value characteristics
    if abs(change) < 1 and 0 <= current_value <= 10:
        # Small, score-like values work well with line charts
        return 'line'
    
    if current_value > 1000 or previous_value > 1000:
        # Large absolute values work well with bar charts
        return 'bar'
    
    if 0 <= current_value <= 100 and 0 <= previous_value <= 100:
        # Percentage-like values work well with pie charts
        return 'pie'
    
    # Default to line chart if no other rules match
    return 'line'

def calculate_article_specific_graph_data(article_category: str, aggregated_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Calculate graph data with visualization types for category-specific metrics."""
    graph_data = {}
    
    # Get metrics from the aggregated data's metrics section
    if not aggregated_data.get('metrics'):
        return graph_data
        
    for metric_name, metric_data in aggregated_data['metrics'].items():
        try:
            characteristics = analyze_metric_characteristics(metric_name, metric_data)
            visualization = determine_best_visualization(characteristics)

            # Add graph data with visualization metadata
            graph_data[metric_name] = {
                'current': metric_data['current'],
                'previous': metric_data['previous'],
                'change': metric_data['change'],
                'change_percentage': metric_data['change_percentage'],
                'visualization': {
                    'type': visualization['type'],
                    'axis_label': visualization['axis_label'],
                    'value_format': visualization['value_format'],
                    'show_points': visualization['show_points'],
                    'stack_type': visualization['stack_type'],
                    'show_labels': visualization['show_labels']
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing metric {metric_name}: {str(e)}")
            continue

    return graph_data

def store_articles(db: Session, articles: List[NewsArticle], date: datetime, organization_id: int):
    """Store articles with visualization metadata."""
    for article in articles:
        # Convert graph data to dictionary format including visualization info
        graph_data_dict = {}
        for metric, data in article.graph_data.items():
            graph_data_dict[metric] = {
                'current': data.current,
                'previous': data.previous,
                'change': data.change,
                'change_percentage': data.change_percentage,
                'visualization': {
                    'type': data.visualization.type if data.visualization else None,
                    'axis_label': data.visualization.axis_label if data.visualization else None,
                    'value_format': data.visualization.value_format if data.visualization else {},
                    'show_points': data.visualization.show_points if data.visualization else False,
                    'stack_type': data.visualization.stack_type if data.visualization else None,
                    'show_labels': data.visualization.show_labels if data.visualization else False
                } if data.visualization else {}
            }

        db_article = Article(
            id=article.id,
            date=date,
            title=article.title,
            content=article.content,
            category=article.category,
            time_period=article.time_period,
            graph_data=graph_data_dict,  # Now contains serializable dictionary
            organization_id=organization_id
        )
        db.add(db_article)
    
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Error storing articles: {str(e)}")
        db.rollback()
        raise

def format_metric_value(metric_name: str, value: Any) -> str:
    """Format metric values based on their type and name patterns."""
    try:
        float_value = float(value)
        
        # Percentage metrics
        if any(term in metric_name.lower() for term in ['rate', 'percentage', 'ratio', 'growth']):
            return f"{float_value:.2f}%"
        
        # Financial metrics
        if any(term in metric_name.lower() for term in ['revenue', 'cost', 'salary', 'spend', 'budget']):
            return f"${float_value:,.2f}"
        
        # Time metrics
        if any(term in metric_name.lower() for term in ['hours', 'time', 'duration']):
            return f"{float_value:.1f} hrs"
        
        # Score/Rating metrics
        if any(term in metric_name.lower() for term in ['score', 'rating', 'satisfaction']):
            return f"{float_value:.2f}"
        
        # Large numbers
        if float_value > 1000:
            return f"{float_value:,.0f}"
        
        # Default format for other numbers
        if float_value.is_integer():
            return f"{int(float_value)}"
        return f"{float_value:.2f}"
        
    except (TypeError, ValueError):
        return str(value)
    
def format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    return metric.replace('_', ' ').title()

def generate_diverse_articles(current_data: Dict[str, Any], previous_data: Dict[str, Any], 
                          time_period: str, end_date: str, org_name: str) -> List[NewsArticle]:
    """Generate articles with visualization-aware graph data."""
    logger.info(f"Generating articles for {org_name}, time_period: {time_period}")
    
    try:
        # Get categories and prepare summary
        categories = get_metric_categories(current_data)
        summary_data = {k: v for k, v in current_data.items() if k != 'graph_data'}
        previous_summary = {k: v for k, v in previous_data.items() if k != 'graph_data'}
        
        summary_parts = [f"Period: {time_period.capitalize()} ending {end_date}"]
        
        # Add metrics to summary with appropriate formatting
        for metric_name, value in summary_data.items():
            try:
                previous_value = previous_summary.get(metric_name, 0)
                change = float(value) - float(previous_value)
                change_percentage = (change / float(previous_value) * 100) if float(previous_value) != 0 else 0
                
                value_str = format_metric_value(metric_name, value)
                
                # Format metric name for display
                display_name = metric_name.replace('_', ' ').title()
                summary_parts.append(
                    f"{display_name}: {value_str} (Change: {change_percentage:.2f}%)"
                )
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logger.error(f"Error processing metric {metric_name}: {str(e)}")
        
        summary = "\n".join(summary_parts)
        category_list = ", ".join(categories.keys())
        
        prompt = f"""
        As a business analyst, generate 5 diverse news articles for {org_name} based on the following {time_period} data summary:
        {summary}
        
        Available metric categories: {category_list}
        
        For each article:
        1. Focus on a specific aspect from one of the available categories and tell a meaningful story.
        2. Provide a title and content that highlights important trends or changes.
        3. Assign one of these categories: {category_list}
        4. Keep each article concise (2-3 sentences).
        5. Reference specific numbers and percentage changes from the summary.
        6. Mention the organization name ({org_name}) in at least one article.
        
        Format your response as a JSON array of objects, each with 'title', 'content', and 'category' keys.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse business news articles from metrics data."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON content and generate articles
        json_content = re.search(r'```json\n(.*?)```', response.choices[0].message.content, re.DOTALL)
        if json_content:
            articles_data = json.loads(json_content.group(1))
        else:
            articles_data = json.loads(response.choices[0].message.content)
        
        # Generate articles with category-specific graph data
        articles = []
        for article in articles_data:
            article_graph_data = calculate_article_specific_graph_data(
                article["category"],
                current_data
            )
            
            articles.append(NewsArticle(
                title=article["title"],
                content=article["content"],
                category=article["category"],
                time_period=time_period,
                graph_data=article_graph_data
            ))
        
        return articles
        
    except Exception as e:
        logger.error(f"Error generating articles: {str(e)}")
        return [NewsArticle(
            title=f"Error in {time_period} article generation",
            content=f"Error generating articles: {str(e)}",
            category="Error",
            time_period=time_period,
            graph_data={}
        )]



def get_data_for_period(db: Session, start_date: datetime, end_date: datetime, org_id: int):
    # Get the organization
    org = db.query(Organization).filter(Organization.id == org_id).first()
    
    # Convert organization name to snake_case for table name
    table_name = org.name.lower().replace(' ', '_')
    
    # Construct the query dynamically
    query = text(f"""
        SELECT 
            SUM(revenue) as total_revenue,
            SUM(units_sold) as total_units_sold,
            AVG(customer_satisfaction) as avg_satisfaction,
            SUM(new_customers) as total_new_customers,
            SUM(marketing_spend) as total_marketing_spend,
            MAX(product) as top_product,
            MAX(location) as top_location
        FROM {table_name}
        WHERE date BETWEEN :start_date AND :end_date
    """)
    
    result = db.execute(query, {
        'start_date': start_date,
        'end_date': end_date
    }).first()

    if result:
        return {
            'total_revenue': result.total_revenue or 0,
            'total_units_sold': result.total_units_sold or 0,
            'avg_satisfaction': result.avg_satisfaction or 0,
            'total_new_customers': result.total_new_customers or 0,
            'total_marketing_spend': result.total_marketing_spend or 0,
            'top_product': result.top_product or 'N/A',
            'top_location': result.top_location or 'N/A'
        }
    return None


def get_stored_articles(db: Session, date: datetime, organization_id: int) -> List[NewsArticle]:
    db_articles = db.query(Article).filter(
        Article.date == date,
        Article.organization_id == organization_id
    ).all()
    
    articles = []
    for article in db_articles:
        # Convert stored graph data back to Pydantic models
        graph_data = {}
        for metric, data in article.graph_data.items():
            # Create Visualization instance if visualization data exists
            visualization = None
            if 'visualization' in data and data['visualization']:
                visualization = Visualization(
                    type=data['visualization'].get('type', 'line'),
                    axis_label=data['visualization'].get('axis_label', 'Value'),
                    value_format=data['visualization'].get('value_format', {}),
                    show_points=data['visualization'].get('show_points', False),
                    stack_type=data['visualization'].get('stack_type'),
                    show_labels=data['visualization'].get('show_labels', False)
                )
            
            # Create GraphData instance
            graph_data[metric] = GraphData(
                current=data['current'],
                previous=data['previous'],
                change=data['change'],
                change_percentage=data['change_percentage'],
                visualization=visualization
            )
        
        # Create NewsArticle instance
        articles.append(NewsArticle(
            id=str(article.id),
            title=article.title,
            content=article.content,
            category=article.category,
            time_period=article.time_period,
            graph_data=graph_data
        ))
    
    return articles

async def verify_headers(
    x_user_id: str = Header(...),
    x_organization_id: str = Header(...),
    x_user_role: str = Header(...),
    authorization: str = Header(...)
):
    """Dependency to verify required headers"""
    if not all([x_user_id, x_organization_id, x_user_role, authorization]):
        raise HTTPException(
            status_code=401,
            detail="Missing required headers"
        )
    return {
        "user_id": x_user_id,
        "org_id": x_organization_id,
        "role": x_user_role,
        "token": authorization
    }

@router.get("/feed", response_model=NewsFeed)
async def get_news_feed(
    request: Request,
    date: str = Query(default=datetime.now().strftime('%Y-%m-%d')),
    db: Session = Depends(get_db)

):
    try:
        # Get user context from headers passed by gateway
        user_id = request.headers.get('X-User-ID')
        current_org_id = request.headers.get('X-Organization-ID')
        role = request.headers.get('X-User-Role')
        
        if not user_id or not current_org_id:
            raise HTTPException(status_code=401, detail="Missing user context")
            
        end_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Get organization name from database
        org = db.query(Organization).filter(Organization.id == int(current_org_id)).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
            
        org_name = org.name
        logger.info(f"Getting news feed for org {current_org_id} ({org_name}) on {date}")
        
        # Get all connections for the organization
        connections_info = get_all_connections_info(current_org_id, db)
        logger.info(f"Found {len(connections_info)} connections for org {current_org_id}")
        
        if not connections_info:
            logger.warning(f"No data sources connected for organization {org_name}")
            return NewsFeed(articles=[
                NewsArticle(
                    id=str(uuid4()),
                    title="Connect Your Data Sources",
                    content=f"Please connect at least one data source for {org_name} to generate insights. Go to Settings > Data Sources to connect your databases.",
                    category="System",
                    time_period="N/A",
                    graph_data={}
                )
            ])

        # Check for data in any source
        has_data = False
        for connection in connections_info:
            connector = None
            try:
                connector = ConnectorFactory.get_connector(
                    connection['source_type'],
                    **connection['params']
                )
                connector.connect()
                
                # Get date column
                if not connection.get('date_column'):
                    db_connection = db.query(models.DataSourceConnection).filter_by(id=connection['connection_id']).first()
                    if db_connection:
                        connection['date_column'] = db_connection.date_column
                
                date_column = connection.get('date_column', 'DATE')
                table_name = connection['table_name']
                
                # Build and log the query before execution
                if connection['source_type'] == 'snowflake':
                    database = connection['params'].get('database')
                    schema = connection['params'].get('schema')
                    
                    # First verify table exists
                    verify_query = f"""
                        SELECT COUNT(*) as EXISTS_FLAG
                        FROM {database}.INFORMATION_SCHEMA.TABLES 
                        WHERE TABLE_SCHEMA = '{schema}'
                        AND TABLE_NAME = '{table_name.upper()}'
                    """
                    verify_result = connector.query(verify_query)
                    if not verify_result or not verify_result[0].get('EXISTS_FLAG', 0):
                        logger.warning(f"Table {table_name} not found in {database}.{schema}")
                        continue
                        
                    # Then check for data
                    data_query = f"""
                        SELECT COUNT(*) as ROW_COUNT 
                        FROM {database}.{schema}."{table_name}"
                        WHERE "{date_column}" <= CURRENT_TIMESTAMP()
                    """
                    logger.info(f"Executing Snowflake query: {data_query}")
                else:
                    data_query = f"""
                        SELECT COUNT(*) as row_count 
                        FROM {table_name}
                        WHERE {date_column} <= CURRENT_DATE
                    """
                    logger.info(f"Executing query: {data_query}")
                
                result = connector.query(data_query)
                row_count = result[0].get('ROW_COUNT' if connection['source_type'] == 'snowflake' else 'row_count', 0)
                
                logger.info(f"Found {row_count} rows in {connection['name']}")
                
                if row_count > 0:
                    has_data = True
                    break
                    
            except Exception as e:
                logger.error(f"Error checking data for source {connection.get('name', '')}: {str(e)}")
                continue
            finally:
                if connector:
                    try:
                        connector.disconnect()
                    except Exception as disconnect_error:
                        logger.error(f"Error disconnecting: {str(disconnect_error)}")
        
        if not has_data:
            logger.warning(f"No data available in any data source for {org_name}")
            return NewsFeed(articles=[
                NewsArticle(
                    id=str(uuid4()),
                    title="No Data Available",
                    content=f"Your data sources for {org_name} are connected but contain no data. Please ensure your data has been uploaded to the connected databases.",
                    category="System",
                    time_period="N/A",
                    graph_data={}
                )
            ])
            
        # Generate articles
        logger.info(f"Generating articles for {org_name}")
        articles = await generate_articles_for_organization(
            db, end_date, current_org_id, org_name
        )
        
        if not articles:
            logger.warning(f"No articles generated for {org_name}")
            return NewsFeed(articles=[
                NewsArticle(
                    id=str(uuid4()),
                    title="Unable to Generate Narratives",
                    content=f"We were unable to generate insights from your data. Please ensure your data sources contain the required columns and data format is correct.",
                    category="System",
                    time_period="N/A",
                    graph_data={}
                )
            ])
        
        logger.info(f"Successfully generated {len(articles)} articles for {org_name}")
        store_articles(db, articles, end_date, current_org_id)
        
        return NewsFeed(articles=articles)
        
    except Exception as e:
        logger.error(f"Error in get_news_feed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while generating narratives: {str(e)}"
        )
    
async def process_gpt_response(
    prompt: str,
    aggregated_data: Dict[str, Any],
    period: str,
    connections_info: List[Dict]
) -> List[NewsArticle]:
    """Process GPT response and create article objects with source information."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a business analyst that generates insightful news articles from metrics data."
                },
                {"role": "user", "content": prompt}
            ]
        )

        # Parse GPT response
        articles_data = json.loads(
            re.search(r'```json\n(.*?)```', response.choices[0].message.content, re.DOTALL).group(1)
            if '```json' in response.choices[0].message.content
            else response.choices[0].message.content
        )

        articles = []
        for article_data in articles_data:
            article_text = f"{article_data['title']} {article_data['content']}".lower()
            article_metrics = {}
            used_sources = set()
            metrics_by_source = {}

            # Extract numbers from the article text to help match metrics
            numbers_in_text = set()
            for match in re.finditer(r"[-+]?\d*\.\d+|\d+", article_text):
                numbers_in_text.add(float(match.group()))

            # Find metrics mentioned in the article based on their values
            for metric_name, metric_data in aggregated_data['metrics'].items():
                current_val = float(metric_data['current'])
                previous_val = float(metric_data['previous'])
                change_pct = float(metric_data['change_percentage'])
                
                # Check if metric values are mentioned in the text
                if (round(current_val, 2) in numbers_in_text or 
                    round(previous_val, 2) in numbers_in_text or 
                    round(change_pct, 1) in numbers_in_text or
                    metric_name.lower().replace('_', ' ') in article_text):
                    
                    # Create visualization for the metric
                    visualization = determine_visualization(metric_name, metric_data)
                    
                    # Add to article metrics
                    article_metrics[metric_name] = GraphData(
                        current=metric_data['current'],
                        previous=metric_data['previous'],
                        change=metric_data['change'],
                        change_percentage=metric_data['change_percentage'],
                        visualization=visualization
                    )

                    # Track source information
                    for source_name, source_values in metric_data['source_values'].items():
                        used_sources.add(source_name)
                        
                        if source_name not in metrics_by_source:
                            metrics_by_source[source_name] = MetricSourceInfo(
                                metrics=[],
                                values={}
                            )
                        
                        if metric_name not in metrics_by_source[source_name].metrics:
                            metrics_by_source[source_name].metrics.append(metric_name)
                            metrics_by_source[source_name].values[metric_name] = source_values

            # Create source info for sources that provided metrics used in this article
            relevant_sources = [
                SourceInfo(
                    id=connection['id'],
                    name=connection['name'],
                    type=connection['source_type']
                )
                for connection in connections_info
                if connection['name'] in used_sources
            ]

            # Create source info if there are relevant sources
            source_info = None
            if relevant_sources:
                source_info = ArticleSourceInfo(
                    sources=relevant_sources,
                    metrics_by_source={
                        source_name: source_data
                        for source_name, source_data in metrics_by_source.items()
                    }
                )

            article = NewsArticle(
                id=str(uuid4()),
                title=article_data['title'],
                content=article_data['content'],
                category=article_data['category'],
                time_period=period,
                graph_data=article_metrics,
                source_info=source_info
            )
            
            articles.append(article)

        return articles

    except Exception as e:
        logger.error(f"Error processing GPT response: {str(e)}")
        raise


def aggregate_metrics_across_sources(sources_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate similar metrics across different data sources."""
    aggregated_metrics = {}
    metric_sources = {}  # Track which sources contribute to each metric
    
    for source_data in sources_data:
        source_name = source_data['source_name']
        source_metrics = source_data['periods']
        
        for period, metrics in source_metrics.items():
            for metric_name, metric_data in metrics.items():
                # Normalize metric name
                normalized_name = normalize_metric_name(metric_name)
                
                if normalized_name not in aggregated_metrics:
                    aggregated_metrics[normalized_name] = {
                        'current': 0,
                        'previous': 0,
                        'sources': set(),
                        'source_values': {}
                    }
                
                # Add source-specific values
                aggregated_metrics[normalized_name]['source_values'][source_name] = {
                    'current': metric_data['current'],
                    'previous': metric_data['previous'],
                    'change': metric_data['change'],
                    'change_percentage': metric_data['change_percentage']
                }
                
                # Update aggregated values
                aggregated_metrics[normalized_name]['current'] += metric_data['current']
                aggregated_metrics[normalized_name]['previous'] += metric_data['previous']
                aggregated_metrics[normalized_name]['sources'].add(source_name)
    
    # Calculate final metrics
    final_metrics = {}
    for metric_name, data in aggregated_metrics.items():
        source_count = len(data['sources'])
        if source_count > 0:
            # Calculate averages for metrics that should be averaged
            should_average = any(term in metric_name.lower() for term in 
                ['average', 'ratio', 'rate', 'satisfaction', 'score'])
            
            current = data['current'] / source_count if should_average else data['current']
            previous = data['previous'] / source_count if should_average else data['previous']
            
            change = current - previous
            change_percentage = (change / previous * 100) if previous != 0 else 0
            
            final_metrics[metric_name] = {
                'current': current,
                'previous': previous,
                'change': change,
                'change_percentage': change_percentage,
                'sources': list(data['sources']),
                'source_count': source_count,
                'source_values': data['source_values']
            }
    
    return final_metrics

def normalize_metric_name(name: str) -> str:
    """Normalize metric names to match similar metrics across sources."""
    # Define common metric patterns and their normalized names
    patterns = {
        r'revenue|income|earnings': 'revenue',
        r'cost|expense|spending': 'cost',
        r'profit|margin': 'profit',
        r'satisfaction|rating|score': 'satisfaction',
        r'users|customers|clients': 'customers',
        r'sales|orders': 'sales',
        r'visits|views|traffic': 'visits'
    }
    
    name_lower = name.lower()
    for pattern, normalized in patterns.items():
        if re.search(pattern, name_lower):
            return normalized
    
    return name_lower

def format_source_breakdown(metric_data: Dict[str, Any]) -> str:
    """Format source-specific breakdown for a metric."""
    breakdown = []
    for source_name, values in metric_data['source_values'].items():
        value_str = format_metric_value(metric_data['metric_name'], values['current'])
        change_str = f"{values['change_percentage']:.1f}%"
        breakdown.append(f"{source_name}: {value_str} (Change: {change_str})")
    
    return "\n".join(breakdown)

def determine_visualization(metric_name: str, metric_data: Dict[str, Any]) -> Visualization:
    """Determine visualization properties for a metric."""
    # Analyze metric characteristics
    characteristics = analyze_metric_characteristics(metric_name, metric_data)
    
    # Get visualization properties
    vis_props = determine_best_visualization(characteristics)
    
    # Create Visualization instance
    return Visualization(
        type=vis_props['type'],
        axis_label=vis_props['axis_label'],
        value_format=vis_props['value_format'],
        show_points=vis_props['show_points'],
        stack_type=vis_props['stack_type'],
        show_labels=vis_props['show_labels']
    )

def prepare_article_summary(aggregated_data: Dict[str, Any], period: str, period_end: datetime) -> str:
    """Prepare detailed summary for article generation."""
    if not aggregated_data or not aggregated_data.get('metrics'):
        return ""
        
    summary_parts = [
        f"Period: {period.capitalize()} ending {period_end.strftime('%Y-%m-%d')}\n"
    ]
    
    # Add overall metrics summary
    summary_parts.append("Overall Metrics:")
    for metric_name, metric_data in aggregated_data['metrics'].items():
        if metric_data['source_count'] > 0:  # Only include metrics with data
            formatted_name = format_metric_name(metric_name)
            value_str = format_metric_value(metric_name, metric_data['current'])
            change_str = f"{metric_data['change_percentage']:.1f}%"
            source_count = metric_data['source_count']
            
            summary_parts.append(
                f"{formatted_name}: {value_str} (Change: {change_str}, Sources: {source_count})"
            )
            
            # Add source breakdown
            if source_count > 1:  # Only show breakdown for multi-source metrics
                summary_parts.append("Source Breakdown:")
                for source_name, source_data in metric_data['source_values'].items():
                    source_value = format_metric_value(metric_name, source_data['current'])
                    source_change = f"{source_data['change_percentage']:.1f}%"
                    summary_parts.append(f"  - {source_name}: {source_value} (Change: {source_change})")
            
            summary_parts.append("")  # Empty line for readability
    
    return "\n".join(summary_parts)

async def generate_articles_for_organization(
    db: Session, 
    end_date: datetime, 
    org_id: int, 
    org_name: str
) -> List[NewsArticle]:
    try:
        connections_info = get_all_connections_info(org_id, db)
        if not connections_info:
            return [NewsArticle(
                id=str(uuid4()),
                title="No Data Sources Connected",
                content=f"Please connect at least one data source for {org_name} to generate insights.",
                category="System",
                time_period="N/A",
                graph_data={},
                source_info=None
            )]

        # Get all discovered metrics for each connection
        all_metrics = {}
        for connection in connections_info:
            metrics = db.query(MetricDefinition).filter(
                MetricDefinition.connection_id == connection['connection_id']
            ).all()
            all_metrics[connection['id']] = metrics

        time_periods = ['daily', 'weekly', 'monthly']
        all_articles = []

        for period in time_periods:
            try:
                # Get aggregated data with source information
                aggregated_data = await fetch_data_from_all_sources(
                    connections_info,
                    end_date,
                    db
                )

                if not aggregated_data:
                    continue

                # Prepare detailed summary including metrics metadata
                summary = prepare_article_summary(aggregated_data, period, end_date)
                if not summary:
                    continue

                # Enhance the prompt with metrics metadata
                metrics_info = []
                for conn_id, metrics in all_metrics.items():
                    for metric in metrics:
                        metrics_info.append({
                            'name': metric.name,
                            'category': metric.category,
                            'business_context': metric.business_context,
                            'visualization_type': metric.visualization_type
                        })

                prompt = f"""
                As a business analyst, generate 5 diverse news articles for {org_name} based on the following {period} data summary:
                {summary}

                Available Metrics Information:
                {json.dumps(metrics_info, indent=2)}

                Consider that the data comes from {len(connections_info)} sources. Generate insights that:
                1. Use the provided metrics in their appropriate business context
                2. Group related metrics by their categories
                3. Suggest appropriate visualizations based on the metric types
                4. Reference specific numbers and changes
                5. Provide context based on the metric's business context
                6. Keep content concise (2-3 sentences)

                Format your response as a JSON array of objects, each with:
                - title: Title of the article
                - content: Article content
                - category: One of the available metric categories
                - metrics_used: Array of metric names used in the article
                - suggested_visualizations: Array of visualization types for the metrics
                """

                # Generate and process articles with source information
                articles = await process_gpt_response(
                    prompt, 
                    aggregated_data, 
                    period, 
                    connections_info
                )
                
                all_articles.extend(articles)

            except Exception as e:
                logger.error(f"Error generating articles for {period} period: {str(e)}")

        if not all_articles:
            return [NewsArticle(
                id=str(uuid4()),
                title="No Insights Available",
                content=f"We couldn't generate insights for {org_name}. Please ensure your data sources contain valid metrics data.",
                category="System",
                time_period="N/A",
                graph_data={},
                source_info=None
            )]

        return all_articles

    except Exception as e:
        logger.error(f"Error in generate_articles: {str(e)}")
        return [NewsArticle(
            id=str(uuid4()),
            title="Error Generating Articles",
            content=f"An error occurred while generating articles: {str(e)}",
            category="Error",
            time_period="N/A",
            graph_data={},
            source_info=None
        )]
    
def generate_pdf_content_single_article(article, chart_image):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                font-size: 24px;
                margin-bottom: 20px;
            }}
            .content {{
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .metadata {{
                font-style: italic;
                color: #7f8c8d;
                margin-top: 15px;
            }}
            .chart {{
                text-align: center;
                margin-top: 30px;
            }}
            .chart img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>{article.title}</h1>
        
        <div class="content">
            <p>{article.content}</p>
            
            <div class="metadata">
                <p><strong>Category:</strong> {article.category}</p>
                <p><strong>Time Period:</strong> {article.time_period}</p>
            </div>
        </div>

        <div class="chart">
            <img src="data:image/png;base64,{chart_image}" alt="Data Chart">
        </div>
    </body>
    </html>
    """
    return html_content

def generate_pdf_fallback_single_article(article, chart_image):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='CenteredTitle', parent=styles['Title'], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='Metadata', parent=styles['BodyText'], fontName='Helvetica-Oblique'))
    
    story = []
    
    # Title
    story.append(Paragraph(article.title, styles['CenteredTitle']))
    story.append(Spacer(1, 0.25*inch))
    
    # Content
    story.append(Paragraph(article.content, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    story.append(Paragraph(f"Category: {article.category}", styles['Metadata']))
    story.append(Paragraph(f"Time Period: {article.time_period}", styles['Metadata']))
    story.append(Spacer(1, 0.3*inch))
    
    # Chart image
    if chart_image:
        try:
            img_data = base64.b64decode(chart_image.split(',')[1] if ',' in chart_image else chart_image)
            img = Image(BytesIO(img_data), width=6*inch, height=3*inch)  # Adjust size as needed
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Error loading chart: {str(e)}", styles['BodyText']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@router.post("/export-pdf-single")
async def export_pdf_single(
    request: PDFRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        article = db.query(Article).filter(Article.id == request.article_id).first()
        
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
            
        # Verify user has access to this article's organization
        if article.organization_id != current_user["current_org_id"]:
            raise HTTPException(status_code=403, detail="You don't have access to this article")
        
        news_article = NewsArticle(
            id=str(article.id),
            title=article.title,
            content=article.content,
            category=article.category,
            time_period=article.time_period,
            graph_data={k: GraphData(**v) if isinstance(v, dict) else v for k, v in article.graph_data.items()}
        )
        
        # Create safe filename first
        safe_filename = "".join(c for c in article.title if c.isalnum() or c in (' ', '-', '_'))
        filename = f"{safe_filename[:50]}.pdf"
        
        html_content = generate_pdf_content_single_article(news_article, request.chart_image)
        
        try:
            # Generate PDF with pdfkit
            pdf_content = pdfkit.from_string(html_content, False)
            
            # Return PDF as binary response
            return Response(
                content=pdf_content,
                media_type='application/pdf',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'application/pdf'
                }
            )
            
        except Exception as pdfkit_error:
            logger.warning(f"pdfkit failed, using fallback method. Error: {str(pdfkit_error)}")
            
            # Try fallback method
            fallback_buffer = generate_pdf_fallback_single_article(news_article, request.chart_image)
            pdf_content = fallback_buffer.getvalue()
            fallback_buffer.close()
            
            return Response(
                content=pdf_content,
                media_type='application/pdf',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'application/pdf'
                }
            )
            
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")
    
def generate_chart(chart_info):
    if not chart_info:
        return None

    plt.figure(figsize=(8, 4))
    data = chart_info['data']
    category = chart_info['category']
    parameter = chart_info['parameter']

    if category == 'Regional':
        plt.pie([data['current'], data['previous']], labels=['Current', 'Previous'], autopct='%1.1f%%')
        plt.title(f'{parameter.replace("_", " ").title()}: Current vs Previous')
    elif category == 'Financial':
        plt.bar(['Previous', 'Current'], [data['previous'], data['current']])
        plt.title(f'{parameter.replace("_", " ").title()}: Previous vs Current')
    else:
        plt.plot(['Previous', 'Current'], [data['previous'], data['current']], marker='o')
        plt.title(f'{parameter.replace("_", " ").title()} Trend')
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return img_str

@router.get("/article/{article_id}/suggested_questions", response_model=List[str])
async def get_article_suggested_questions(
    article_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate suggested questions using GPT based on article content and metrics.
    
    Args:
        article_id (str): UUID of the article
        db (Session): Database session
        current_user (dict): Current user information
        
    Returns:
        List[str]: List of suggested questions related to the article content
    """
    try:
        # Validate user
        user = current_user.get("user")
        if not user:
            raise HTTPException(status_code=400, detail="Invalid user information")

        # Get the article
        article = db.query(Article).filter(Article.id == article_id).first()
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        # Extract metrics data and format it for the prompt
        metrics_data = article.graph_data if article.graph_data else {}
        metrics_summary = []
        
        for metric_name, metric_data in metrics_data.items():
            try:
                current_val = float(metric_data.get('current', 0))
                prev_val = float(metric_data.get('previous', 0))
                change_pct = float(metric_data.get('change_percentage', 0))
                display_name = metric_name.replace('_', ' ').title()
                
                metrics_summary.append(
                    f"{display_name}: Current: {current_val:,.2f}, "
                    f"Previous: {prev_val:,.2f}, "
                    f"Change: {change_pct:+.1f}%"
                )
            except (ValueError, TypeError) as e:
                continue

        # Create a detailed prompt for GPT
        prompt = f"""As a business analyst, generate upto 8 insightful follow-up questions for this article which should be less than 7 words:

Article Title: {article.title}
Category: {article.category}
Time Period: {article.time_period}
Content: {article.content}

Metrics Analysis:
{chr(10).join(metrics_summary)}

Generate questions that:
1. Address significant metric changes
2. Explore business implications
3. Consider the time period ({article.time_period}) context
4. Relate to the article category ({article.category})
5. Suggest potential actions or strategies
6. Focus on root causes and correlations
7. Consider future implications
8. Connect to broader business context

Format your response as a JSON array of strings, each string being a question.
Include a mix of questions about performance analysis, strategic implications, and actionable insights.
"""

        try:
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business analyst helping generate insightful follow-up questions about business metrics and performance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            # Extract questions from response
            response_text = response.choices[0].message.content.strip()
            
            # Handle different response formats
            try:
                if response_text.startswith('```json'):
                    # Extract JSON from code block
                    json_str = response_text.split('```json')[1].split('```')[0]
                    gpt_questions = json.loads(json_str)
                else:
                    # Try direct JSON parsing
                    gpt_questions = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract questions line by line
                gpt_questions = [
                    line.strip().strip('"-,')
                    for line in response_text.split('\n')
                    if line.strip().endswith('?')
                ]

            # Add rule-based questions as backup
            rule_based_questions = []
            
            # Add questions for significant metric changes
            for metric_name, metric_data in metrics_data.items():
                try:
                    change_pct = float(metric_data.get('change_percentage', 0))
                    if abs(change_pct) > 10:
                        display_name = metric_name.replace('_', ' ').title()
                        direction = "increase" if change_pct > 0 else "decrease"
                        rule_based_questions.append(
                            f"What factors contributed to the {abs(change_pct):.1f}% {direction} in {display_name}?"
                        )
                except (ValueError, TypeError):
                    continue

            # Add time-period specific question
            period = article.time_period.lower()
            if 'monthly' in period:
                rule_based_questions.append("How do these monthly trends compare to our quarterly targets?")
            elif 'quarterly' in period:
                rule_based_questions.append("What are the implications for our annual objectives?")
            elif 'weekly' in period:
                rule_based_questions.append("How do these weekly patterns align with monthly trends?")

            # Add category-specific question
            category = article.category.lower()
            if 'financial' in category:
                rule_based_questions.append("What is the projected impact on our annual financial goals?")
            elif 'customer' in category:
                rule_based_questions.append("How can we leverage these insights to improve customer experience?")
            elif 'performance' in category:
                rule_based_questions.append("What operational improvements could maintain this performance?")

            # Combine and deduplicate questions
            all_questions = gpt_questions + rule_based_questions
            unique_questions = list(dict.fromkeys(all_questions))  # Preserve order while deduplicating

            # Return top 5 questions, prioritizing GPT-generated ones
            return unique_questions[:5]

        except Exception as e:
            logger.error(f"Error with GPT question generation: {str(e)}")
            # Fallback to rule-based questions
            return [
                f"What are the key drivers behind these {article.category.lower()} results?",
                f"How do these trends compare to our historical {article.time_period.lower()} performance?",
                "What strategic actions should we consider based on these insights?",
                "How do these results affect our business objectives?",
                "What additional analysis would help us better understand these trends?"
            ]

    except Exception as e:
        logger.exception(f"Error generating article questions: {str(e)}")
        return [
            f"What insights can we draw from this {article.category.lower()} analysis?",
            "What actions should we take based on these findings?",
            "How does this impact our current strategies?",
            "What stakeholders should be informed about these results?",
            "What additional data would help us better understand these trends?"
        ]