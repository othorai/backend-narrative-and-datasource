# connectors/mssql_connector.py

import pyodbc
import logging
from app.connectors.base import BaseConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MSSQLConnector(BaseConnector):
    def __init__(self, host, username, password, database, port=1433):
        super().__init__()  # Call parent constructor
        self.source_type = 'mssql'  # Set the source type
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.port = int(port) if port else 1433
        self.connection = None
        self.driver = '{ODBC Driver 18 for SQL Server}'  # Default driver

    def connect(self):
        """Establish connection to MSSQL with error handling."""
        try:
            logger.info(f"Attempting to connect to MSSQL database {self.database} as user {self.username}")
            connection_string = (
                f"DRIVER={self.driver};"
                f"SERVER={self.host};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
                "TrustServerCertificate=yes;"
                "Connect Timeout=30;" 
                "LoginTimeout=30;"   
            )
            
            self.connection = pyodbc.connect(connection_string)
            logger.info("Successfully connected to MSSQL")
            
        except pyodbc.Error as e:
            logger.error(f"MSSQL connection error: {str(e)}")
            if self.connection:
                self.connection.close()
                self.connection = None
                
            if "Login failed" in str(e):
                raise ValueError(f"Authentication failed for user '{self.username}'. Please verify credentials.")
            elif "Could not connect to server" in str(e):
                raise ValueError(f"Could not connect to database server at {self.host}:{self.port}. Please verify connection details.")
            else:
                raise ValueError(f"Database connection failed: {str(e)}")

    def disconnect(self):
        """Safely close the connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                logger.info("MSSQL connection closed")
        except Exception as e:
            logger.error(f"Error closing MSSQL connection: {str(e)}")

    def query(self, query_string, params=None):
        """Execute query with error handling and automatic reconnection."""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()  # Get cursor from our mocked connection
            
            # Execute query with or without parameters
            if params:
                cursor.execute(query_string, params)
            else:
                cursor.execute(query_string)
                
            # Only try to get column names if we have results
            if cursor.description:
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
                
                # Convert rows to dictionaries
                result = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    result.append(row_dict)
                return result
                
            return []
                    
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query_string}")
            logger.error(f"Params: {params}")
            raise ValueError(f"Query execution failed: {str(e)}")

    def insert(self, table, data):
        """Insert data with error handling."""
        try:
            with self.connection.cursor() as cursor:
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['?' for _ in data])
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(query, list(data.values()))
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Insert operation failed: {str(e)}")
            raise ValueError(f"Failed to insert data: {str(e)}")

    def update(self, table, data, condition):
        """Update data with error handling."""
        try:
            with self.connection.cursor() as cursor:
                set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
                query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
                cursor.execute(query, list(data.values()))
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Update operation failed: {str(e)}")
            raise ValueError(f"Failed to update data: {str(e)}")

    def delete(self, table, condition):
        """Delete data with error handling."""
        try:
            with self.connection.cursor() as cursor:
                query = f"DELETE FROM {table} WHERE {condition}"
                cursor.execute(query)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Delete operation failed: {str(e)}")
            raise ValueError(f"Failed to delete data: {str(e)}")