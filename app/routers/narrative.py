#routers/narrative.py
from fastapi import APIRouter, Depends, Query, HTTPException, Request, Header, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uuid
import calendar
from app.services.DynamicDataAnalysisService import DynamicAnalysisService
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
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

# Load environment variablesx
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

def store_articles(db: Session, articles: List[NewsArticle], date: datetime, org_id: int):
    """Store articles with visualization metadata and suggested questions."""
    for article in articles:
        try:
            # Convert graph data to dictionary format including visualization info
            graph_data_dict = {}
            for metric, data in article.graph_data.items():
                graph_data_dict[metric] = {
                    'current': data.current,
                    'previous': data.previous,
                    'change': data.change,
                    'change_percentage': data.change_percentage,
                    'visualization': {
                        'type': data.visualization.type if data.visualization else 'line',
                        'axis_label': data.visualization.axis_label if data.visualization else 'Value',
                        'value_format': data.visualization.value_format if data.visualization else {},
                        'show_points': data.visualization.show_points if data.visualization else True,
                        'stack_type': data.visualization.stack_type if data.visualization else None,
                        'show_labels': data.visualization.show_labels if data.visualization else True
                    } if data.visualization else {}
                }
                
            db_article = Article(
                id=uuid.UUID(article.id),  # Convert string ID to UUID
                date=date,
                title=article.title,
                content=article.content,
                category=article.category,
                time_period=article.time_period,
                context=getattr(article, 'context', None),  # Get context if it exists
                graph_data=graph_data_dict,
                organization_id=org_id
            )

            # Check if article already exists
            existing_article = db.query(Article).filter(Article.id == db_article.id).first()
            if existing_article:
                # Update existing article
                for key, value in vars(db_article).items():
                    if not key.startswith('_'):
                        setattr(existing_article, key, value)
            else:
                # Add new article
                db.add(db_article)

            logger.info(f"Storing article: {article.id} for org: {org_id}")
            
        except Exception as e:
            logger.error(f"Error storing article {article.id}: {str(e)}")
            continue
    
    try:
        db.commit()
        logger.info(f"Successfully stored {len(articles)} articles")
    except Exception as e:
        logger.error(f"Error committing articles to database: {str(e)}")
        db.rollback()
        raise

    return True

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

def get_week_of_month(date: datetime) -> int:
    """Get week number within the month (1-5)."""
    first_day = date.replace(day=1)
    dom = date.day
    adjusted_dom = dom + first_day.weekday()
    week_of_month = (adjusted_dom - 1) // 7 + 1
    return week_of_month

def should_generate_feed(period: str, target_date: datetime) -> Tuple[bool, Tuple[datetime, datetime]]:
    """
    Determine if feed should be generated and return the appropriate date range.
    Returns tuple of (should_generate, (start_date, end_date))
    """
    today = datetime.now()
    
    if period == "daily":
        return True, (target_date, target_date)
        
    elif period == "weekly":
        # Get the week of month for current date
        current_week = get_week_of_month(today)
        target_week = get_week_of_month(target_date)
        
        # Calculate start and end dates for the week
        week_start = target_date - timedelta(days=target_date.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Generate if it's the current week or previous week
        if target_week == current_week or target_week == current_week - 1:
            return True, (week_start, week_end)
            
        return False, (None, None)
        
    elif period == "monthly":
        today_day = today.day
        target_month = target_date.replace(day=1)
        
        if today_day <= 15:
            # First half of month - report on previous month's second half
            prev_month_mid = (target_month - timedelta(days=1)).replace(day=16)
            prev_month_end = target_month - timedelta(days=1)
            return True, (prev_month_mid, prev_month_end)
        else:
            # Second half of month - report on current month's first half
            month_start = target_month
            month_mid = target_month.replace(day=15)
            return True, (month_start, month_mid)
    
    return False, (None, None)

def format_period_data(
    period: str,
    target_date: datetime,
    start_date: datetime,
    end_date: datetime
) -> Tuple[str, str]:
    """
    Format period display text and context.
    Returns tuple of (display_text, context)
    """
    if period == "daily":
        display = target_date.strftime("%d-%m-%Y")
        context = display
        
    elif period == "weekly":
        week_num = get_week_of_month(target_date)
        display = f"Week {week_num}"
        context = display
        
    else:  # monthly
        if target_date.day <= 15:
            # First half reporting on previous month's second half
            prev_month = (target_date.replace(day=1) - timedelta(days=1))
            display = f"{prev_month.strftime('%B %Y')} (16th-{prev_month.day})"
            context = prev_month.strftime('%B')
        else:
            # Second half reporting on current month's first half
            display = f"{target_date.strftime('%B %Y')} (1st-15th)"
            context = target_date.strftime('%B')
    
    return display, context

def get_period_date_range(date: datetime, period: str) -> tuple[datetime, datetime]:
    """Get start and end dates for the specified period."""
    if period == "daily":
        return date, date
    elif period == "weekly":
        # Get Monday and Sunday of the week
        start = date - timedelta(days=date.weekday())
        end = start + timedelta(days=6)
        return start, end
    else:  # monthly
        # First and last day of the month
        start = date.replace(day=1)
        end = date.replace(
            day=calendar.monthrange(date.year, date.month)[1]
        )
        return start, end

async def check_existing_feed(db: Session, org_id: int, date: datetime, period: str) -> Optional[List[NewsArticle]]:
    """Check if feed already exists in database."""
    try:
        start_date, end_date = get_period_date_range(date, period)
        
        # Query existing articles
        articles = db.query(Article).filter(
            Article.organization_id == org_id,
            Article.date >= start_date,
            Article.date <= end_date,
            Article.time_period == period
        ).all()
        
        if articles:
            return [
                NewsArticle(
                    id=str(article.id),
                    title=article.title,
                    content=article.content,
                    category=article.category,
                    time_period=f"{article.time_period} ({format_period_date(article.date, article.time_period)})",
                    graph_data=article.graph_data,
                    context=get_period_context(article.date, article.time_period),
                )
                for article in articles
            ]
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking existing feed: {str(e)}")
        return None

def format_period_date(date: datetime, period: str) -> str:
    """Format date based on time period."""
    if period == "daily":
        return date.strftime("%d-%m-%Y")
    elif period == "weekly":
        return f"Week {get_week_of_month(date)}"
    else:  # monthly
        if date.day <= 15:
            prev_month = (date.replace(day=1) - timedelta(days=1))
            return f"{prev_month.strftime('%B %Y')} (16th-{prev_month.day})"
        else:
            return f"{date.strftime('%B %Y')} (1st-15th)"

def get_period_context(date: datetime, period: str) -> str:
    """Get context for given period and date."""
    if period == "daily":
        return date.strftime("%d-%m-%Y")
    elif period == "weekly":
        return f"Week {get_week_of_month(date)}"
    else:  # monthly
        if date.day <= 15:
            prev_month = (date.replace(day=1) - timedelta(days=1))
            return prev_month.strftime('%B')
        else:
            return date.strftime('%B')
                
async def get_user_context(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-ID"),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role")
) -> Dict[str, Any]:
    
    if settings.STAGE == "DEV":
        # Use default values in DEV mode
        context = {
            "user_id": settings.DEFAULT_USER_ID,
            "org_id": int(settings.DEFAULT_ORG_ID),
            "role": settings.DEFAULT_USER_ROLE
        }
        logger.info(f"Using DEV mode defaults: {context}")
        return context
    
    # Production mode - require headers
    if not all([x_user_id, x_organization_id, x_user_role]):
        logger.error(f"Missing headers. Received: User ID: {x_user_id}, Org ID: {x_organization_id}, Role: {x_user_role}")
        raise HTTPException(status_code=401, detail="Missing required headers")
    
    try:
        return {
            "user_id": x_user_id,
            "org_id": int(x_organization_id),
            "role": x_user_role
        }
    except ValueError as e:
        logger.error(f"Error parsing organization ID: {e}")
        raise HTTPException(status_code=400, detail="Invalid organization ID format")

def get_dynamic_scope(period: str, target_date: datetime) -> Tuple[str, str]:
    """
    Convert period and target date into appropriate scope for DynamicAnalysisService.
    Returns tuple of (scope, resolution).
    """
    today = datetime.now()
    
    if period == "daily":
        days_diff = (today - target_date).days
        if days_diff == 0:
            return "today", "daily"
        elif days_diff == 1:
            return "yesterday", "daily"
        else:
            return f"last_{days_diff}_days", "daily"
            
    elif period == "weekly":
        # Get the start of current week (Monday)
        current_week_start = today - timedelta(days=today.weekday())
        
        # Calculate week number of target date
        target_week_num = int(target_date.strftime('%U'))
        current_week_num = int(today.strftime('%U'))
        
        if target_week_num == current_week_num:
            return "this_week", "weekly"
        elif target_week_num == current_week_num - 1:
            return "last_week", "weekly"
        else:
            return "prior_week", "weekly"
            
    elif period == "monthly":
        # Get first day of current month
        current_month_start = today.replace(day=1)
        
        if today.day <= 15:
            # First half of month, look at previous month
            return "last_month", "monthly"
        else:
            # Second half of month, look at current month to date
            return "month_to_date", "monthly"
    
    return "this_year", "monthly"  # default fallback

async def get_existing_narratives(db: Session, date: datetime, org_id: int) -> Optional[List[NewsArticle]]:
    """Get existing narratives for the given date and organization."""
    try:
        # Query existing articles for the date and organization
        articles = db.query(Article).filter(
            Article.date == date.date(),  # Use .date() to ignore time component
            Article.organization_id == org_id
        ).all()
        
        if not articles:
            return None
            
        # Convert to NewsArticle format
        narratives = []
        for article in articles:
            # Convert stored graph data to GraphData format
            graph_data = {}
            for metric_name, data in article.graph_data.items():
                visualization = None
                if 'visualization' in data:
                    visualization = Visualization(
                        type=data['visualization'].get('type', 'line'),
                        axis_label=data['visualization'].get('axis_label', 'Value'),
                        value_format=data['visualization'].get('value_format', {}),
                        show_points=data['visualization'].get('show_points', True),
                        stack_type=data['visualization'].get('stack_type'),
                        show_labels=data['visualization'].get('show_labels', True)
                    )

                graph_data[metric_name] = GraphData(
                    current=float(data.get('current', 0)),
                    previous=float(data.get('previous', 0)),
                    change=float(data.get('change', 0)),
                    change_percentage=float(data.get('change_percentage', 0)),
                    visualization=visualization
                )

            narratives.append(NewsArticle(
                id=str(article.id),
                title=article.title,
                content=article.content,
                category=article.category,
                time_period=article.time_period,
                context=article.context,
                graph_data=graph_data
            ))
            
        logger.info(f"Found {len(narratives)} existing narratives for date {date.date()} and org {org_id}")
        return narratives
        
    except Exception as e:
        logger.error(f"Error fetching existing narratives: {str(e)}")
        return None
    
@router.get("/feed", response_model=NewsFeed)
async def get_news_feed(
    date: str = Query(default=datetime.now().strftime('%Y-%m-%d')),
    db: Session = Depends(get_db),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get news feed for all applicable periods."""
    try:
        logger.info(f"Processing feed request - User Context: {user_context}")
        
        org_id = user_context['org_id']
        
        # Validate organization exists
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            error_msg = f"Organization not found: {org_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
            
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Check for existing narratives first
        existing_narratives = await get_existing_narratives(db, target_date, org_id)
        if existing_narratives:
            logger.info(f"Returning {len(existing_narratives)} existing narratives")
            return NewsFeed(articles=existing_narratives)
            
        # If no existing narratives, proceed with generation
        all_articles = []
        
        # Initialize dynamic analysis service
        analysis_service = DynamicAnalysisService()
        
        # Process each period in priority order
        periods = ["daily", "weekly", "monthly"]
        
        for period in periods:
            try:
                should_generate, (start_date, end_date) = should_generate_feed(period, target_date)
                
                if not should_generate:
                    logger.info(f"Skipping {period} feed generation for {date}")
                    continue
                
                logger.info(f"Generating {period} feed for period {start_date} to {end_date}")
                
                # Get appropriate scope for DynamicAnalysisService
                if period == "daily":
                    scope = "today"
                elif period == "weekly":
                    scope = "this_week" if get_week_of_month(target_date) == get_week_of_month(datetime.now()) else "last_week"
                else:  # monthly
                    scope = "last_month" if target_date.day <= 15 else "month_to_date"
                
                # Get period-specific data using dynamic analysis
                period_data = await analysis_service.analyze_metrics(
                    db=db,
                    org_id=org_id,
                    scope=scope,
                    resolution=period,
                    forecast=False
                )
                
                if period_data and period_data.get('metrics'):
                    # Create prompts based on analyzed data
                    prompt = await create_dynamic_prompt(
                        org_name=org.name,
                        period_data=period_data,
                        period=period,
                        target_date=target_date
                    )
                    
                    # Generate articles using analyzed data
                    new_articles = await process_gpt_response(
                        prompt=prompt,
                        aggregated_data=period_data,
                        period=period,
                        connections_info=get_all_connections_info(org_id, db)
                    )
                    
                    if new_articles:
                        # Format period display and context
                        display_text, context = format_period_data(
                            period=period,
                            target_date=target_date,
                            start_date=start_date,
                            end_date=end_date
                        )

                        # Get connections info once for all articles
                        connections_info = get_all_connections_info(org_id, db)

                        for article in new_articles:
                            article.time_period = f"{period} ({display_text})"
                            article.context = context

                            # Create source info for metrics in the article
                            used_sources = set()
                            metrics_by_source = {}

                            # Process each metric to track source information
                            for metric_name, metric_data in period_data.get('metrics', {}).items():
                                if metric_name in article.graph_data:
                                    # Track source information for this metric
                                    for source_info in metric_data.get('sources', []):
                                        source_name = source_info.get('name', 'Unknown')
                                        used_sources.add(source_name)
                                        
                                        if source_name not in metrics_by_source:
                                            metrics_by_source[source_name] = MetricSourceInfo(
                                                metrics=[],
                                                values={}
                                            )
                                        
                                        if metric_name not in metrics_by_source[source_name].metrics:
                                            metrics_by_source[source_name].metrics.append(metric_name)
                                            metrics_by_source[source_name].values[metric_name] = {
                                                'current': source_info.get('current', 0),
                                                'previous': source_info.get('previous', 0),
                                                'change': source_info.get('change', {}).get('absolute', 0),
                                                'change_percentage': source_info.get('change', {}).get('percentage', 0)
                                            }

                            # Create source info if we found any sources
                            if used_sources:
                                relevant_sources = [
                                    SourceInfo(
                                        id=connection['id'],
                                        name=connection['name'],
                                        type=connection['source_type']
                                    )
                                    for connection in connections_info
                                    if connection['name'] in used_sources
                                ]
                                
                                if relevant_sources:
                                    article.source_info = ArticleSourceInfo(
                                        sources=relevant_sources,
                                        metrics_by_source=metrics_by_source
                                    )
                        
                        all_articles.extend(new_articles)
                        logger.info(f"Generated {len(new_articles)} articles for {period} period")
                
            except Exception as e:
                logger.error(f"Error processing {period} feed: {str(e)}")
                continue
        
        if not all_articles:
            return NewsFeed(articles=[
                NewsArticle(
                    id=str(uuid4()),
                    title="No Articles Available",
                    content="No articles could be generated for the specified date.",
                    category="System",
                    context="N/A",
                    time_period="N/A",
                    graph_data={}
                )
            ])
        
        # Store all generated articles for future use
        store_articles(db, all_articles, target_date, org_id)
        
        # Sort articles by priority
        all_articles.sort(
            key=lambda x: periods.index(x.time_period.split()[0])
        )
        
        logger.info(f"Successfully generated feed with {len(all_articles)} articles")
        return NewsFeed(articles=all_articles)
        
    except Exception as e:
        error_msg = f"Error in get_news_feed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

async def create_dynamic_prompt(
    org_name: str,
    period_data: Dict[str, Any],
    period: str,
    target_date: datetime
) -> str:
    """Create dynamic prompt based on analyzed data."""
    try:
        # Get appropriate date labels based on period
        if period == "daily":
            date_label = target_date.strftime("%d-%m-%Y")
            comparison = "previous day"
        elif period == "weekly":
            week_num = int(target_date.strftime('%U'))
            date_label = f"Week {week_num} ({target_date.strftime('%B %Y')})"
            comparison = "previous week"
        else:  # monthly
            if target_date.day <= 15:
                date_label = f"{target_date.strftime('%B %Y')} (1st-15th)"
            else:
                last_day = calendar.monthrange(target_date.year, target_date.month)[1]
                date_label = f"{target_date.strftime('%B %Y')} (16th-{last_day})"
            comparison = "previous period"

        # Organize metrics by category
        metrics_by_category = {}
        for metric_name, metric_data in period_data.get('metrics', {}).items():
            category = metric_data.get('category', 'Other')
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append({
                'name': metric_name,
                'data': metric_data
            })

        # Build category-specific insights
        category_insights = []
        for category, metrics in metrics_by_category.items():
            metric_summaries = []
            for metric in metrics:
                current_val = metric['data'].get('current_value', 0)
                change_pct = metric['data'].get('change', {}).get('percentage', 0)
                trend = "increased" if change_pct > 0 else "decreased" if change_pct < 0 else "remained stable"
                
                metric_summaries.append(
                    f"{metric['name']}: {current_val:,.2f} ({trend} by {abs(change_pct):.1f}% from {comparison})"
                )
            
            if metric_summaries:
                category_insights.append(f"{category}:\n" + "\n".join(metric_summaries))

        # Create comprehensive prompt
        prompt = f"""
        As a business analyst, generate comprehensive insights for {org_name} based on the following {period} data:

        Period: {period.capitalize()} ({date_label})
        
        Metrics Analysis by Category:
        {chr(10).join(category_insights)}

        Key Analysis Requirements:
        1. Focus on significant changes and their implications
        2. Consider {period} trends and patterns
        3. Analyze relationships between different metrics
        4. Provide specific, actionable recommendations
        5. Consider both positive and negative trends
        6. Highlight opportunities for improvement
        7. Compare performance with {comparison}
        8. Consider the business context for each metric

        Generate multiple detailed articles that:
        1. Cover each significant metric or trend
        2. Explain the business impact
        3. Provide actionable insights
        4. Reference specific numbers and changes
        5. Suggest concrete next steps
        6. Consider cross-category relationships
        7. Look at both short and long-term implications

        Format your response as a JSON array of objects, each with:
        - 'title': Clear, metric-focused title
        - 'content': 2-3 sentences of analysis and recommendations
        - 'category': One of the available categories shown above

        Generate articles for all significant metrics and trends in the data, with special attention to:
        - Metrics with significant changes (>10% change)
        - Interrelated metrics that tell a broader story
        - Metrics that indicate important business trends
        - Areas needing immediate attention or action

        Do not limit the number of articles - create an article for each meaningful insight.
        """

        return prompt

    except Exception as e:
        logger.error(f"Error creating prompt: {str(e)}")
        raise
    
async def process_gpt_response(
    prompt: str,
    aggregated_data: Dict[str, Any],
    period: str,
    connections_info: List[Dict]
) -> List[NewsArticle]:
    """
    Process GPT response and create article objects with source information.
    Integrates with DynamicAnalysisService for better metric handling.
    """
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
        target_date = datetime.now()
        seen_insights = set()  # Track unique insights

        for article_data in articles_data:
            # Generate unique key for this insight
            insight_key = f"{article_data['title']}_{article_data['category']}"
            if insight_key in seen_insights:
                continue
            
            seen_insights.add(insight_key)
            
            article_text = f"{article_data['title']} {article_data['content']}".lower()
            article_metrics = {}
            used_sources = set()
            metrics_by_source = {}

            # Match metrics based on analyzed data
            for metric_name, metric_data in aggregated_data.get('metrics', {}).items():
                try:
                    # Get current and previous values from the analyzed data
                    current_val = float(metric_data.get('current_value', 0))
                    previous_val = float(metric_data.get('previous_value', 0))
                    change_pct = float(metric_data.get('change', {}).get('percentage', 0))
                    
                    # Extract numbers from the article text
                    numbers_in_text = set()
                    for match in re.finditer(r"[-+]?\d*\.?\d+%?", article_text):
                        try:
                            # Handle percentage values
                            num_str = match.group().rstrip('%')
                            numbers_in_text.add(float(num_str))
                        except ValueError:
                            continue

                    # Check if metric is mentioned or values are referenced
                    metric_mentioned = (
                        metric_name.lower().replace('_', ' ') in article_text or
                        any(term in article_text for term in metric_name.lower().split('_')) or
                        round(current_val, 2) in numbers_in_text or
                        round(previous_val, 2) in numbers_in_text or
                        round(change_pct, 1) in numbers_in_text
                    )

                    if metric_mentioned:
                        # Get visualization from analyzed data
                        visualization = Visualization(
                            type=metric_data.get('visualization_type', 'line'),
                            axis_label=metric_data.get('axis_label', 'Value'),
                            value_format=metric_data.get('value_format', {
                                'type': 'number',
                                'decimal_places': 2,
                                'prefix': '',
                                'suffix': '',
                                'use_grouping': True
                            }),
                            show_points=metric_data.get('show_points', True),
                            stack_type=metric_data.get('stack_type'),
                            show_labels=metric_data.get('show_labels', True)
                        )

                        # Create graph data with analyzed information
                        article_metrics[metric_name] = GraphData(
                            current=current_val,
                            previous=previous_val,
                            change=float(metric_data.get('change', {}).get('absolute', 0)),
                            change_percentage=change_pct,
                            visualization=visualization
                        )

                        # Track source information
                        for source_info in metric_data.get('sources', []):
                            source_name = source_info.get('name', 'Unknown')
                            used_sources.add(source_name)
                            
                            if source_name not in metrics_by_source:
                                metrics_by_source[source_name] = MetricSourceInfo(
                                    metrics=[],
                                    values={}
                                )
                            
                            if metric_name not in metrics_by_source[source_name].metrics:
                                metrics_by_source[source_name].metrics.append(metric_name)
                                metrics_by_source[source_name].values[metric_name] = {
                                    'current': source_info.get('current', 0),
                                    'previous': source_info.get('previous', 0),
                                    'change': source_info.get('change', {}).get('absolute', 0),
                                    'change_percentage': source_info.get('change', {}).get('percentage', 0)
                                }

                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing metric {metric_name}: {str(e)}")
                    continue

            # Create source info for relevant sources
            source_info = None
            if used_sources:
                relevant_sources = [
                    SourceInfo(
                        id=connection['id'],
                        name=connection['name'],
                        type=connection['source_type']
                    )
                    for connection in connections_info
                    if connection['name'] in used_sources
                ]
                
                if relevant_sources:
                    source_info = ArticleSourceInfo(
                        sources=relevant_sources,
                        metrics_by_source=metrics_by_source
                    )

            # Create article with enhanced information
            article = NewsArticle(
                id=str(uuid4()),
                title=article_data['title'],
                content=article_data['content'],
                category=article_data['category'],
                time_period=f"{period} ({format_period_date(target_date, period)})",
                context=get_period_context(target_date, period),
                graph_data=article_metrics,
                source_info=source_info
            )
            
            articles.append(article)

        return articles

    except Exception as e:
        logger.error(f"Error processing GPT response: {str(e)}", exc_info=True)
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

async def generate_articles_for_organization(
    db: Session, 
    target_date: datetime, 
    org_id: int, 
    period: str
) -> List[NewsArticle]:
    """Generate comprehensive articles for an organization across different time periods."""
    try:
        connections_info = get_all_connections_info(org_id, db)
        if not connections_info:
            return [NewsArticle(
                id=str(uuid4()),
                title="No Data Sources Connected",
                content=f"Please connect at least one data source to generate insights.",
                category="System",
                time_period=f"{period} ({format_period_date(target_date, period)})",
                graph_data={},
                source_info=None
            )]

        # Calculate date range based on period
        start_date, end_date = get_period_date_range(target_date, period)
        logger.info(f"Generating {period} articles for period: {start_date} to {end_date}")

        # Get period-specific data
        current_data = await fetch_data_from_all_sources(
            connections_info,
            end_date,
            db,
            start_date=start_date
        )

        # Calculate previous period
        previous_period = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)  # Approximate month
        }
        period_delta = previous_period[period]
        prev_end = start_date
        prev_start = prev_end - period_delta

        previous_data = await fetch_data_from_all_sources(
            connections_info,
            prev_end,
            db,
            start_date=prev_start
        )

        if not current_data or not previous_data:
            return [NewsArticle(
                id=str(uuid4()),
                title=f"No Data Available for {period.capitalize()} Analysis",
                content=f"No data available for the {period} period ending {end_date.strftime('%Y-%m-%d')}.",
                category="System",
                time_period=f"{period} ({format_period_date(target_date, period)})",
                graph_data={},
                source_info=None
            )]

        # Get organization name
        org = db.query(Organization).filter(Organization.id == org_id).first()
        org_name = org.name if org else "Organization"

        # Period-specific context
        period_contexts = {
            "daily": {
                "timeframe": "day",
                "comparison": "yesterday",
                "trend_scope": "daily operations"
            },
            "weekly": {
                "timeframe": "week",
                "comparison": "last week",
                "trend_scope": "weekly performance"
            },
            "monthly": {
                "timeframe": "month",
                "comparison": "last month",
                "trend_scope": "monthly trends"
            }
        }
        context = period_contexts[period]

        # Create detailed summary
        summary = prepare_article_summary(
            current_data,
            period,
            end_date
        )

        # Create comprehensive prompt
        prompt = f"""
        As a business analyst, generate comprehensive insights for {org_name} based on {context['timeframe']} data:
        
        Period: {period.capitalize()} ({format_period_date(target_date, period)})
        Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        
        Data Summary:
        {summary}
        
        Generate multiple detailed articles covering:
        1. Key Performance Indicators:
           - Financial metrics and their implications
           - Operational efficiency indicators
           - Revenue and cost analysis
        
        2. Human Resources & Training:
           - Employee performance metrics
           - Training and development indicators
           - Workforce efficiency measures
        
        3. Customer Metrics:
           - Customer satisfaction and retention
           - Service quality indicators
           - Customer behavior patterns
        
        4. Operational Excellence:
           - Process efficiency metrics
           - Resource utilization
           - Productivity indicators
        
        5. Growth & Development:
           - Market performance indicators
           - Development metrics
           - Innovation measures
        
        For each article:
        1. Focus on trends specific to this {period} period
        2. Compare with {context['comparison']}'s performance
        3. Provide actionable insights and recommendations
        4. Reference specific numbers and percentage changes
        5. Consider the impact on {context['trend_scope']}
        6. Highlight both positive and negative trends
        7. Suggest strategic actions based on the data
        
        Format your response as a JSON array of objects, each with:
        - 'title': Clear, metric-focused title
        - 'content': 2-3 sentences of analysis and recommendations
        - 'category': Relevant business category
        
        Do not limit the number of articles - generate an article for each significant metric or trend in the data.
        """

        # Generate articles with period-specific data
        articles = await process_gpt_response(
            prompt,
            current_data,
            period,
            connections_info
        )

        # Add period-specific formatting to articles
        formatted_articles = []
        for article in articles:
            formatted_article = article
            formatted_article.time_period = f"{period} ({format_period_date(target_date, period)})"
            formatted_articles.append(formatted_article)

        logger.info(f"Generated {len(formatted_articles)} articles for {period} period")
        return formatted_articles

    except Exception as e:
        logger.error(f"Error generating {period} articles: {str(e)}", exc_info=True)
        return [NewsArticle(
            id=str(uuid4()),
            title=f"Error Generating {period.capitalize()} Articles",
            content=f"An error occurred while generating articles: {str(e)}",
            category="Error",
            time_period=f"{period} ({format_period_date(target_date, period)})",
            graph_data={},
            source_info=None
        )]

def prepare_article_summary(
    aggregated_data: Dict[str, Any], 
    period: str,
    period_end: datetime
) -> str:
    """Prepare period-specific summary for article generation."""
    if not aggregated_data or not aggregated_data.get('metrics'):
        return ""
        
    period_names = {
        "daily": "day",
        "weekly": "week",
        "monthly": "month"
    }
    
    summary_parts = [
        f"Analysis for the {period_names[period]} ending {period_end.strftime('%Y-%m-%d')}\n"
    ]
    
    # Add metrics summary with period context
    metrics_added = set()
    for metric_name, metric_data in aggregated_data['metrics'].items():
        if metric_data['source_count'] > 0 and metric_name not in metrics_added:
            formatted_name = format_metric_name(metric_name)
            value_str = format_metric_value(metric_name, metric_data['current'])
            change_str = f"{metric_data['change_percentage']:.1f}%"
            source_count = metric_data['source_count']
            
            summary_parts.append(
                f"{formatted_name}: {value_str} (Change from previous {period_names[period]}: {change_str})"
            )
            
            # Add source breakdown for multiple sources
            if source_count > 1:
                summary_parts.append("Source Breakdown:")
                for source_name, source_data in metric_data['source_values'].items():
                    source_value = format_metric_value(metric_name, source_data['current'])
                    source_change = f"{source_data['change_percentage']:.1f}%"
                    summary_parts.append(f"  - {source_name}: {source_value} (Change: {source_change})")
            
            metrics_added.add(metric_name)
            summary_parts.append("")  # Empty line for readability
    
    return "\n".join(summary_parts)
    
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
    Get or generate suggested questions for an article.
    """
    try:
        # Validate user
        user = current_user.get("user")
        if not user:
            raise HTTPException(status_code=400, detail="Invalid user information")

        # Get the article
        article = db.query(Article).filter(Article.id == article_id).first()
        if not article:
            logger.error(f"Article not found with ID: {article_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Article not found with ID: {article_id}"
            )

        # Check for cached questions - make sure to handle None case
        if article.suggested_questions and isinstance(article.suggested_questions, list):
            logger.info(f"Returning cached questions for article {article_id}")
            return article.suggested_questions

        # If no cached questions, generate new ones
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

        # Create prompt for GPT
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

Format your response as a JSON array of strings, each string being a question."""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business analyst helping generate insightful follow-up questions about business metrics and performance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            response_text = response.choices[0].message.content.strip()
            
            try:
                if response_text.startswith('```json'):
                    json_str = response_text.split('```json')[1].split('```')[0]
                    questions = json.loads(json_str)
                else:
                    questions = json.loads(response_text)
            except json.JSONDecodeError:
                questions = [
                    line.strip().strip('"-,')
                    for line in response_text.split('\n')
                    if line.strip().endswith('?')
                ]

            # Add rule-based questions
            rule_based_questions = []
            
            # Add metric-based questions
            for metric_name, metric_data in metrics_data.items():
                try:
                    change_pct = float(metric_data.get('change_percentage', 0))
                    if abs(change_pct) > 10:
                        display_name = metric_name.replace('_', ' ').title()
                        direction = "increase" if change_pct > 0 else "decrease"
                        rule_based_questions.append(
                            f"What factors contributed to {display_name} {direction}?"
                        )
                except (ValueError, TypeError):
                    continue

            # Add period-specific questions
            period = article.time_period.lower()
            if 'monthly' in period:
                rule_based_questions.append("How do monthly trends affect quarterly targets?")
            elif 'quarterly' in period:
                rule_based_questions.append("What are annual objective implications?")
            elif 'weekly' in period:
                rule_based_questions.append("How do weekly patterns affect month?")

            # Add category-specific questions
            category = article.category.lower()
            if 'financial' in category:
                rule_based_questions.append("Impact on annual financial goals?")
            elif 'customer' in category:
                rule_based_questions.append("How to improve customer experience?")
            elif 'performance' in category:
                rule_based_questions.append("What improvements maintain performance?")

            # Combine and deduplicate questions
            all_questions = questions + rule_based_questions
            unique_questions = list(dict.fromkeys(all_questions))
            final_questions = unique_questions[:5]

            # Store the questions as a proper JSON array
            try:
                # Begin a nested transaction
                db.begin_nested()
                
                # Update the article with the new questions
                article.suggested_questions = final_questions
                
                # Commit the nested transaction
                db.commit()
                
                logger.info(f"Successfully stored {len(final_questions)} questions for article {article_id}")
            except Exception as db_error:
                db.rollback()
                logger.error(f"Database error storing questions: {str(db_error)}")
                # Continue execution to at least return the questions even if storage failed
            
            return final_questions

        except Exception as e:
            logger.error(f"Error with GPT question generation: {str(e)}")
            fallback_questions = [
                f"What drives changes in {article.category}?",
                f"How does {article.time_period} compare historically?",
                "What strategic actions are needed?",
                "How do results affect objectives?",
                "What further analysis is needed?"
            ]
            
            # Store fallback questions
            try:
                db.begin_nested()
                article.suggested_questions = fallback_questions
                db.commit()
                logger.info(f"Stored fallback questions for article {article_id}")
            except Exception as db_error:
                db.rollback()
                logger.error(f"Database error storing fallback questions: {str(db_error)}")
            
            return fallback_questions

    except Exception as e:
        logger.exception(f"Error generating article questions: {str(e)}")
        default_questions = [
            "What insights can we draw?",
            "What actions should we take?",
            "How does this impact strategy?",
            "Who needs to know this?",
            "What additional data needed?"
        ]
        
        # Try to store default questions
        try:
            db.begin_nested()
            article.suggested_questions = default_questions
            db.commit()
            logger.info(f"Stored default questions for article {article_id}")
        except Exception as db_error:
            db.rollback()
            logger.error(f"Database error storing default questions: {str(db_error)}")
            
        return default_questions