from pymongo import MongoClient
from app.connectors.base import BaseConnector
import logging
from bson import ObjectId
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBConnector(BaseConnector):
    def __init__(self, host: str, username: str = None, password: str = None, database: str = None, port: int = 27017):
        super().__init__()
        self.source_type = 'mongodb'
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB with error handling."""
        try:
            # Create the MongoDB connection URI
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}"
            else:
                uri = f"mongodb://{self.host}:{self.port}"
            
            logger.info(f"Attempting to connect to MongoDB database {self.database}")
            self.connection = MongoClient(uri)
            self.db = self.connection[self.database]
            
            # Test connection by executing a simple command
            self.db.command('ping')
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            if self.connection:
                self.connection.close()
                self.connection = None
            raise ValueError(f"Failed to establish MongoDB connection: {str(e)}")

    def disconnect(self):
        """Safely close the MongoDB connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.db = None
                logger.info("MongoDB connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {str(e)}")
            raise ValueError(f"Failed to close MongoDB connection: {str(e)}")

    def query(self, collection: str, query: Dict = None, projection: Dict = None) -> List[Dict]:
        """Execute a MongoDB query."""
        if not self.connection:
            self.connect()
        
        try:
            collection_obj = self.db[collection]
            cursor = collection_obj.find(query or {}, projection or {})
            
            # Convert cursor to list of documents
            results = []
            for doc in cursor:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
                if any(isinstance(value, datetime) for value in doc.values()):
                    # Found at least one date field, document is good for analysis
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Collection: {collection}")
            logger.error(f"Query: {query}")
            raise ValueError(f"Query execution failed: {str(e)}")

    def get_schema_info(self, collection_name: str):
        """Get schema information for MongoDB collection by sampling documents"""
        try:
            collection = self.db[collection_name]
            
            # Sample some documents directly
            sample_docs = list(collection.find().limit(100))
            
            # Analyze field types across documents
            field_types = {}
            date_fields = set()
            
            for doc in sample_docs:
                for field, value in doc.items():
                    if isinstance(value, datetime):
                        date_fields.add(field)
                    current_type = type(value).__name__
                    if field not in field_types:
                        field_types[field] = set()
                    field_types[field].add(current_type)
            
            # Format results similar to SQL schema
            schema_info = []
            for field, types in field_types.items():
                type_str = '/'.join(sorted(types))
                schema_info.append({
                    'column_name': field,
                    'data_type': type_str,
                    'is_date': field in date_fields
                })
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting MongoDB schema: {str(e)}")
            raise ValueError(f"Failed to get MongoDB schema: {str(e)}")

    def detect_date_column(self, collection_name: str) -> Optional[str]:
        """
        Detect the primary date column in a MongoDB collection
        """
        try:
            schema_info = self.get_schema_info(collection_name)
            date_columns = [
                col['column_name'] for col in schema_info 
                if col['is_date']
            ]
            
            # Prioritize common date field names
            priority_names = ['dateOfJoining', 'date', 'timestamp', 'created_at', 'createdAt', 'hire_date']
            for name in priority_names:
                if name in date_columns:
                    return name
                    
            # If no priority match, return the first date column found
            return date_columns[0] if date_columns else None
            
        except Exception as e:
            logger.error(f"Error detecting date column: {str(e)}")
            return None

    def insert(self, collection: str, data: Dict):
        """Insert a document into MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            result = collection_obj.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Insert operation failed: {str(e)}")
            raise ValueError(f"Failed to insert data: {str(e)}")

    def update(self, collection: str, query: Dict, data: Dict, upsert: bool = False):
        """Update documents in MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            result = collection_obj.update_many(
                query,
                {'$set': data},
                upsert=upsert
            )
            return {
                'matched_count': result.matched_count,
                'modified_count': result.modified_count,
                'upserted_id': str(result.upserted_id) if result.upserted_id else None
            }
        except Exception as e:
            logger.error(f"Update operation failed: {str(e)}")
            raise ValueError(f"Failed to update data: {str(e)}")

    def delete(self, collection: str, query: Dict):
        """Delete documents from MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            result = collection_obj.delete_many(query)
            return {
                'deleted_count': result.deleted_count
            }
        except Exception as e:
            logger.error(f"Delete operation failed: {str(e)}")
            raise ValueError(f"Failed to delete data: {str(e)}")

    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        """Execute an aggregation pipeline on a MongoDB collection."""
        try:
            collection_obj = self.db[collection]
            results = []
            for doc in collection_obj.aggregate(pipeline):
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            raise ValueError(f"Failed to execute aggregation: {str(e)}")

    def verify_collection_exists(self, collection: str) -> bool:
        """Verify that a collection exists in the current database."""
        try:
            return collection in self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Error verifying collection existence: {str(e)}")
            return False

    def get_collection_stats(self, collection: str) -> Dict:
        """Get statistics about a collection."""
        try:
            return self.db.command('collStats', collection)
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}