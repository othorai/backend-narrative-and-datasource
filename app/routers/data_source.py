# routers/data_source.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.connectors.connector_factory import ConnectorFactory
from app.utils.auth import get_current_user
from app.services.metric_discovery import MetricDiscoveryService
from app.models.models import User, Organization, DataSourceConnection,MetricDefinition
import app.models.models as models
from decimal import Decimal
from app.schemas.schemas import DataSourceConnection, DataSourceConnectionResponse
import logging 
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.utils.database import get_db
from openai import OpenAI
import os
from uuid import uuid4
from app.services.DateColumnDetection import DateColumnDetection
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DataSourceConnection(BaseModel):
    source_type: str
    name: str  # Added name field for identifying different sources
    host: Optional[str] = None
    user: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    port: Optional[int] = None
    table_name: str
    credentials_file: Optional[str] = None
    spreadsheet_id: Optional[str] = None
    security_token: Optional[str] = None
    domain: Optional[str] = 'login'
    account: Optional[str] = None
    warehouse: Optional[str] = None
    schema: Optional[str] = None

def get_all_connections_info(org_id: int, db: Session) -> List[Dict]:
    """Get all connection info for an organization."""
    connections = db.query(models.DataSourceConnection).filter(
        models.DataSourceConnection.organization_id == org_id
    ).all()
    
    if connections:
        # Update last used timestamp for all connections
        for connection in connections:
            connection.updated_at = datetime.utcnow()
        db.commit()
        
        return [conn.to_dict() for conn in connections]
    return []



def discover_table_metrics(connector: Any, table_name: str) -> Dict[str, str]:
    """Dynamically discover available metrics from table columns."""
    try:
        # Get column information
        schema_query = f"""
            SELECT 
                column_name, 
                data_type
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """
        
        columns = connector.query(schema_query)
        metrics = {}
        
        # Common metric patterns and their SQL aggregations
        metric_patterns = {
            'revenue': 'SUM',
            'cost': 'SUM',
            'price': 'AVG',
            'spend': 'SUM',
            'count': 'COUNT',
            'quantity': 'SUM',
            'amount': 'SUM',
            'sales': 'SUM',
            'satisfaction': 'AVG',
            'rating': 'AVG',
            'score': 'AVG',
            'customers': 'COUNT',
            'users': 'COUNT',
            'visits': 'SUM',
            'conversion': 'AVG',
            'time': 'AVG',
            'duration': 'AVG'
        }
        
        numeric_types = {'integer', 'decimal', 'numeric', 'double precision', 'real'}
        
        for col in columns:
            col_name = col['column_name'].lower()
            data_type = col['data_type'].lower()
            
            # Skip date/timestamp columns and non-numeric types
            if 'date' in data_type or 'time' in data_type or data_type not in numeric_types:
                continue
            
            # Determine appropriate aggregation based on column name
            agg_function = None
            for pattern, agg in metric_patterns.items():
                if pattern in col_name:
                    agg_function = agg
                    break
            
            if agg_function:
                metrics[col_name] = f"{agg_function}({col_name})"
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error discovering metrics: {str(e)}")
        return {}

async def fetch_data_from_all_sources(
    connections_info: List[Dict], 
    end_date: datetime, 
    db: Session,
    start_date: Optional[datetime] = None
) -> Dict:
    """
    Fetch and aggregate data from all connected data sources.
    
    Args:
        connections_info: List of connection information dictionaries
        end_date: End date for data fetching
        db: Database session
        start_date: Optional start date for data fetching (defaults to end_date - 1 day)
    """
    logger.info(f"Fetching data from multiple sources for period: {start_date} to {end_date}")
    
    # If no start_date provided, default to previous day for daily comparison
    if start_date is None:
        start_date = end_date - timedelta(days=1)
        
    aggregated_result = {
        'metrics': {},
        'graph_data': {},
        'sources_data': [],
        'period_metrics': {}
    }
    
    for connection_info in connections_info:
        try:
            # Pass both start and end dates to fetch_data_from_source
            source_data = await fetch_data_from_source(
                connection_info=connection_info,
                end_date=end_date,
                db=db,
                start_date=start_date
            )
            
            if not source_data or not source_data.get('graph_data'):
                continue

            source_name = connection_info.get('name', 'Unknown Source')
            logger.info(f"Processing data from source: {source_name}")
            
            # Store source-specific data
            aggregated_result['sources_data'].append({
                'source_name': source_name,
                'periods': {
                    'current': {
                        'start': start_date.strftime('%Y-%m-%d'),
                        'end': end_date.strftime('%Y-%m-%d')
                    },
                    'previous': {
                        'start': (start_date - timedelta(days=(end_date - start_date).days)).strftime('%Y-%m-%d'),
                        'end': start_date.strftime('%Y-%m-%d')
                    },
                    'metrics': source_data.get('graph_data', {})
                }
            })
            
            # Aggregate metrics across sources
            for metric_name, metric_data in source_data.get('graph_data', {}).items():
                if metric_name not in aggregated_result['metrics']:
                    aggregated_result['metrics'][metric_name] = {
                        'current': 0,
                        'previous': 0,
                        'change': 0,
                        'change_percentage': 0,
                        'source_count': 0,
                        'source_values': {}
                    }
                
                # Update metric data
                metric = aggregated_result['metrics'][metric_name]
                metric['current'] += metric_data['current']
                metric['previous'] += metric_data['previous']
                metric['source_count'] += 1
                metric['source_values'][source_name] = {
                    'current': metric_data['current'],
                    'previous': metric_data['previous'],
                    'change': metric_data['change'],
                    'change_percentage': metric_data['change_percentage']
                }
            
        except Exception as e:
            logger.error(f"Error processing source {connection_info.get('name')}: {str(e)}")
            continue

    # Calculate final aggregated metrics
    for metric_name, metric_data in aggregated_result['metrics'].items():
        try:
            source_count = metric_data['source_count']
            if source_count > 0:
                # Handle averaging for specific metric types
                should_average = any(term in metric_name.lower() for term in 
                    ['average', 'avg', 'rate', 'ratio', 'satisfaction', 'score'])
                
                if should_average:
                    metric_data['current'] /= source_count
                    metric_data['previous'] /= source_count
                
                # Calculate changes
                metric_data['change'] = metric_data['current'] - metric_data['previous']
                metric_data['change_percentage'] = (
                    (metric_data['change'] / metric_data['previous'] * 100)
                    if metric_data['previous'] != 0 else 0
                )
                
                aggregated_result['graph_data'][metric_name] = {
                    'current': metric_data['current'],
                    'previous': metric_data['previous'],
                    'change': metric_data['change'],
                    'change_percentage': metric_data['change_percentage'],
                    'source_count': source_count,
                    'source_values': metric_data['source_values']
                }
                
        except Exception as e:
            logger.error(f"Error calculating final metrics for {metric_name}: {str(e)}")
            continue
    
    if not aggregated_result['metrics']:
        logger.warning("No valid metrics found from any source")
        return None
        
    return aggregated_result

def verify_data_exists(connector: Any, table_name: str, start_date: str, end_date: str) -> bool:
    """Verify that data exists for the given date range."""
    try:
        query = f"""
            SELECT EXISTS (
                SELECT 1 
                FROM {table_name} 
                WHERE date BETWEEN %s AND %s
            ) as has_data
        """
        result = connector.query(query, (start_date, end_date))
        return result[0]['has_data'] if result else False
    except Exception as e:
        logger.error(f"Error verifying data existence: {str(e)}")
        return False

async def fetch_data_from_source(
    connection_info: Dict, 
    end_date: datetime, 
    db: Session,
    start_date: Optional[datetime] = None
) -> Dict:
    """
    Fetch data from a single source for a specific time period.
    
    Args:
        connection_info: Connection information dictionary
        end_date: End date for data fetching
        db: Database session
        start_date: Start date for data fetching (defaults to end_date - 1 day)
    """
    logger.info(f"Fetching data for period: {start_date} to {end_date}")
    logger.info(f"Connection info: {json.dumps({k: v for k, v in connection_info.items() if k not in ['connection_params', 'params', 'password']})}")
    
    if not connection_info:
        raise ValueError("No connection information available")
    
    # If no start_date provided, default to previous day
    if start_date is None:
        start_date = end_date - timedelta(days=1)
        
    # Calculate previous period dates
    period_length = (end_date - start_date).days
    previous_end = start_date
    previous_start = previous_end - timedelta(days=period_length)
    
    try:
        # Get date column and validate
        date_column = connection_info.get('date_column')
        if not date_column:
            connection = db.query(models.DataSourceConnection).filter_by(id=connection_info['connection_id']).first()
            if connection:
                date_column = connection.date_column
            if not date_column:
                raise ValueError("No date column found for connection")

        logger.info(f"Using date column: {date_column} for table {connection_info['table_name']}")
        
        # Get metrics for this connection
        metrics = db.query(MetricDefinition).filter(
            models.MetricDefinition.connection_id == connection_info['connection_id']
        ).all()
        
        logger.info(f"Found {len(metrics)} metrics for connection")
        for metric in metrics:
            logger.info(f"Metric details - Name: {metric.name}, Calculation: {metric.calculation}")
        
        if not metrics:
            raise ValueError("No metrics found for this connection")
        
        # Create connector instance
        connector = ConnectorFactory.get_connector(
            connection_info['source_type'],
            **connection_info['params']
        )
        
        try:
            connector.connect()
            table_name = connection_info['table_name']
            
            # Time periods for comparison
            current_end = end_date
            current_start = end_date - timedelta(days=30)
            previous_end = current_start
            previous_start = previous_end - timedelta(days=30)
            
            result = {}
            graph_data = {}
            
            if connection_info['source_type'] == 'snowflake':
                database = connection_info['params'].get('database')
                schema = connection_info['params'].get('schema')
                
                # First verify table exists
                verify_query = f"""
                    SELECT COUNT(*) as "TABLE_EXISTS"
                    FROM {database}.INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = '{schema}'
                    AND TABLE_NAME = '{table_name}'
                """
                verify_result = connector.query(verify_query)
                logger.info(f"Table verification result: {verify_result}")
                
                for metric in metrics:
                    try:
                        # Log metric details
                        logger.info(f"\nProcessing metric: {metric.name}")
                        logger.info(f"Original calculation: {metric.calculation}")
                        
                        # Handle DISTINCT case specially
                        if 'DISTINCT' in metric.calculation.upper():
                            calculation = metric.calculation.upper().replace(
                                'COUNT(DISTINCT employeeid)',
                                'COUNT(DISTINCT "EMPLOYEEID")'
                            )
                        else:
                            # Convert calculation for Snowflake
                            calculation = (metric.calculation.upper()
                                        .replace('AVG(', 'AVG("')
                                        .replace('COUNT(', 'COUNT("')
                                        .replace('SUM(', 'SUM("')
                                        .replace('MAX(', 'MAX("')
                                        .replace('MIN(', 'MIN("')
                                        .replace('SALARY)', 'SALARY")')
                                        .replace('EMPLOYEEID)', 'EMPLOYEEID")')
                                        .replace('NAME)', 'NAME")')
                                        .replace('DEPARTMENT)', 'DEPARTMENT")')
                                        )
                        
                        logger.info(f"Modified calculation: {calculation}")
                        
                        # Build query using string interpolation for dates
                        current_query = f"""
                            SELECT 
                                COALESCE({calculation}, 0) AS "METRIC_VALUE"
                            FROM "{database}"."{schema}"."{table_name}"
                            WHERE "{date_column}" BETWEEN '{current_start.strftime('%Y-%m-%d')}' AND '{current_end.strftime('%Y-%m-%d')}'
                        """
                        
                        prev_query = f"""
                            SELECT 
                                COALESCE({calculation}, 0) AS "METRIC_VALUE"
                            FROM "{database}"."{schema}"."{table_name}"
                            WHERE "{date_column}" BETWEEN '{previous_start.strftime('%Y-%m-%d')}' AND '{previous_end.strftime('%Y-%m-%d')}'
                        """
                        
                        # Log the queries
                        logger.info(f"Current period query: {current_query}")
                        logger.info(f"Previous period query: {prev_query}")
                        
                        # Execute queries
                        current_result = connector.query(current_query)
                        logger.info(f"Current period result: {current_result}")
                        
                        previous_result = connector.query(prev_query)
                        logger.info(f"Previous period result: {previous_result}")
                        
                        # Extract values
                        current_value = float(current_result[0].get('METRIC_VALUE', 0)) if current_result else 0
                        previous_value = float(previous_result[0].get('METRIC_VALUE', 0)) if previous_result else 0
                        
                        # Calculate changes
                        change = current_value - previous_value
                        change_percentage = (change / previous_value * 100) if previous_value != 0 else 0
                        
                        # Store results
                        metric_key = metric.name.lower().replace(" ", "_")
                        result[metric_key] = current_value
                        graph_data[metric_key] = {
                            "current": current_value,
                            "previous": previous_value,
                            "change": change,
                            "change_percentage": change_percentage,
                            "source": connection_info.get('name', 'Unknown Source'),
                            "category": metric.category,
                            "visualization_type": metric.visualization_type,
                            "confidence_score": metric.confidence_score,
                            "business_context": metric.business_context
                        }
                        
                        logger.info(f"Successfully processed metric {metric.name}: current={current_value}, previous={previous_value}")
                        
                    except Exception as e:
                        logger.error(f"Error processing metric {metric.name}: {str(e)}", exc_info=True)
                        continue
            
            else:
                # Handle PostgreSQL and other databases
                for metric in metrics:
                    try:
                        # Log metric details
                        logger.info(f"\nProcessing metric: {metric.name}")
                        logger.info(f"Original calculation: {metric.calculation}")
                        
                        # Build query for current period
                        current_query = f"""
                            WITH period_data AS (
                                SELECT {metric.calculation} as metric_value
                                FROM {table_name}
                                WHERE {date_column} BETWEEN %s AND %s
                            )
                            SELECT COALESCE((SELECT metric_value FROM period_data), 0) as metric_value
                        """
                        
                        # Execute for current period
                        current_params = (
                            current_start.strftime('%Y-%m-%d'),
                            current_end.strftime('%Y-%m-%d')
                        )
                        logger.info(f"Current period query: {current_query}")
                        logger.info(f"Current period params: {current_params}")
                        current_result = connector.query(current_query, current_params)
                        logger.info(f"Current period result: {current_result}")
                        
                        # Execute for previous period
                        previous_params = (
                            previous_start.strftime('%Y-%m-%d'),
                            previous_end.strftime('%Y-%m-%d')
                        )
                        logger.info(f"Previous period params: {previous_params}")
                        previous_result = connector.query(current_query, previous_params)
                        logger.info(f"Previous period result: {previous_result}")
                        
                        # Extract values
                        try:
                            current_value = float(current_result[0]['metric_value'] if current_result else 0)
                            previous_value = float(previous_result[0]['metric_value'] if previous_result else 0)
                            
                            # Calculate changes
                            change = current_value - previous_value
                            change_percentage = (change / previous_value * 100) if previous_value != 0 else 0
                            
                            # Store results
                            metric_key = metric.name.lower().replace(" ", "_")
                            result[metric_key] = current_value
                            graph_data[metric_key] = {
                                "current": current_value,
                                "previous": previous_value,
                                "change": change,
                                "change_percentage": change_percentage,
                                "source": connection_info.get('name', 'Unknown Source'),
                                "category": metric.category,
                                "visualization_type": metric.visualization_type,
                                "confidence_score": metric.confidence_score,
                                "business_context": metric.business_context
                            }
                            
                            logger.info(f"Successfully processed metric {metric.name}: current={current_value}, previous={previous_value}")
                            
                        except (KeyError, ValueError, TypeError) as e:
                            logger.error(f"Error extracting values for metric {metric.name}: {str(e)}")
                            logger.error(f"Current result structure: {current_result}")
                            logger.error(f"Previous result structure: {previous_result}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing metric {metric.name}: {str(e)}", exc_info=True)
                        continue
            
            if not result:
                raise ValueError("No metrics could be calculated")
            
            result['graph_data'] = graph_data
            return result
            
        finally:
            if connector:
                try:
                    connector.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}", exc_info=True)
        raise

@router.post("/connect")
async def connect_to_data_source(
    connection: DataSourceConnection,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user = current_user["user"]
        org_id = current_user["current_org_id"]
        
        logger.info(f"Attempting to connect to {connection.source_type} for organization: {org_id}")
        
        # Extract connection parameters
        connector_params = connection.dict(exclude={'table_name', 'source_type', 'name', 'date_column'})
        table_name = connection.table_name
        source_type = connection.source_type
        source_name = connection.name

        # Clean up connector parameters
        if source_type == 'postgresql':
            # Ensure username is properly set
            if connection.username:
                connector_params['username'] = connection.username
            elif connection.user:
                connector_params['username'] = connection.user
                
            # Remove None or empty values but preserve explicit values
            connector_params = {k: v for k, v in connector_params.items() 
                             if v not in (None, '') or k in ('username', 'password', 'host', 'database')}
            
            # Set default port if not provided
            if 'port' not in connector_params or not connector_params['port']:
                connector_params['port'] = 5432

            # Log connection parameters (excluding password)
            log_params = {k: v if k != 'password' else '****' for k, v in connector_params.items()}
            logger.info(f"PostgreSQL connection parameters: {log_params}")

            # Verify required parameters
            required_params = ['host', 'username', 'password', 'database']
            missing_params = [param for param in required_params if param not in connector_params or not connector_params[param]]
            
            if missing_params:
                raise ValueError(f"Missing required connection parameters: {', '.join(missing_params)}")

        # Test connection
        try:
            connector = ConnectorFactory.get_connector(
                source_type,
                **connector_params
            )
            
            connector.connect()
            
            # Initialize OpenAI client
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Detect suitable date column using OpenAI
            date_detector = DateColumnDetection(openai_client)
            date_column = await date_detector.detect_date_column(connector, table_name)
            
            if not date_column:
                raise HTTPException(
                    status_code=400,
                    detail="No suitable date column found for time series analysis"
                )
            
            # Store in database
            db_connection = models.DataSourceConnection(
                id=str(uuid4()),
                organization_id=org_id,
                name=source_name,
                source_type=source_type,
                connection_params=connector_params,
                table_name=table_name,
                date_column=date_column
            )
            
            db.add(db_connection)
            db.commit()
            
            # Discover metrics
            metric_service = MetricDiscoveryService(openai_client)
            await metric_service.discover_metrics(db_connection.id, db)
            
            # Update organization
            org = db.query(Organization).filter(Organization.id == org_id).first()
            org.data_source_connected = True
            db.commit()
            
            connector.disconnect()
            
            return {
                "message": f"Successfully connected to {source_type}",
                "connection_id": db_connection.id,
                "date_column": date_column
            }
            
        except ValueError as ve:
            logger.error(f"Connection validation failed: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Connection test failed: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Connection error: {str(e)}")

@router.delete("/connections/{connection_id}")
async def remove_data_source(
    connection_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        org_id = current_user["current_org_id"]
        
        # Find and delete the connection
        connection = db.query(models.DataSourceConnection).filter(
            models.DataSourceConnection.id == connection_id,
            models.DataSourceConnection.organization_id == org_id
        ).first()
        
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")
            
        # Delete associated metrics
        db.query(models.MetricDefinition).filter(
            models.MetricDefinition.connection_id == connection_id
        ).delete()
        
        # Delete the connection
        db.delete(connection)
        
        # Check if this was the last connection
        remaining_connections = db.query(models.DataSourceConnection).filter(
            models.DataSourceConnection.organization_id == org_id
        ).count()
        
        if remaining_connections == 0:
            # Update organization status
            org = db.query(Organization).filter(Organization.id == org_id).first()
            org.data_source_connected = False
        
        db.commit()
        
        return {"message": "Data source removed successfully"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error removing data source: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/organization/{org_id}/data-sources")
async def list_organization_data_sources(
    org_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if user has access to this organization
        user = current_user["user"]
        org = db.query(Organization).filter(Organization.id == org_id).first()
        
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        if not user.is_admin and org not in user.organizations:
            raise HTTPException(status_code=403, detail="Not authorized to access this organization's data sources")
        
        # Get data sources from database
        data_sources = db.query(models.DataSourceConnection).filter(
            models.DataSourceConnection.organization_id == org_id
        ).all()
        
        # Format the response
        formatted_sources = []
        for source in data_sources:
            data_source = {
                "id": source.id,
                "name": source.name,
                "source_type": source.source_type,
                "table_name": source.table_name,
                "connected": True,
                "connection_details": {
                    "host": source.connection_params.get('host'),
                    "database": source.connection_params.get('database'),
                    "username": source.connection_params.get('username'),
                }
            }
            formatted_sources.append(data_source)
            
        return {
            "organization_id": org_id,
            "organization_name": org.name,
            "data_sources": formatted_sources,
            "is_connected": org.data_source_connected
        }
        
    except Exception as e:
        logger.error(f"Error listing data sources: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
@router.put("/connections/{connection_id}")
async def update_data_source(
    connection_id: str,
    connection: DataSourceConnection,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        org_id = current_user["current_org_id"]
        
        # Find existing connection
        db_connection = db.query(models.DataSourceConnection).filter(
            models.DataSourceConnection.id == connection_id,
            models.DataSourceConnection.organization_id == org_id
        ).first()
        
        if not db_connection:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        # Update the name
        db_connection.name = connection.name
        
        # Update other fields if needed
        if connection.table_name:
            db_connection.table_name = connection.table_name
        
        if connection.source_type:
            db_connection.source_type = connection.source_type
            
        if any(val for val in connection.dict().values()):
            db_connection.connection_params = {
                **db_connection.connection_params,
                **{k: v for k, v in connection.dict().items() if v is not None}
            }
            
        db.commit()
        
        return {
            "message": "Data source updated successfully",
            "connection_id": connection_id,
            "name": db_connection.name
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating data source: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))