
# SQL/NoSQL database integration
import logging
from typing import Any
from sqlalchemy import create_engine
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class SQLDatabase:
    """
    SQL database integration using SQLAlchemy.
    """
    def __init__(self, uri: str):
        """
        Initialize the SQL database connection.
        Args:
            uri (str): SQLAlchemy database URI.
        """
        try:
            self.engine = create_engine(uri)
            logger.info(f"Connected to SQL database at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to SQL database: {e}")
            self.engine = None

class NoSQLDatabase:
    """
    NoSQL database integration using MongoDB.
    """
    def __init__(self, uri: str):
        """
        Initialize the NoSQL database connection.
        Args:
            uri (str): MongoDB URI.
        """
        try:
            self.client = MongoClient(uri)
            logger.info(f"Connected to MongoDB at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
