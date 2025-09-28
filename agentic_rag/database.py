
# SQL/NoSQL database integration
import logging
from typing import Any
from sqlalchemy import create_engine
from pymongo import MongoClient

logger = logging.getLogger(__name__)


from sqlalchemy import text


from datetime import datetime

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
            self._ensure_table()
        except Exception as e:
            logger.error(f"Failed to connect to SQL database: {e}")
            self.engine = None

    def _ensure_table(self):
        """Create table for Q&A if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS qa_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            model TEXT,
            valid BOOLEAN,
            timestamp TEXT
        );
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_table_sql))

    def save_qa(self, question: str, answer: str, model: str, valid: bool):
        """Save a Q&A record to the SQL database."""
        insert_sql = text("""
            INSERT INTO qa_records (question, answer, model, valid, timestamp)
            VALUES (:question, :answer, :model, :valid, :timestamp)
        """)
        with self.engine.connect() as conn:
            conn.execute(insert_sql, {
                'question': question,
                'answer': answer,
                'model': model,
                'valid': valid,
                'timestamp': datetime.utcnow().isoformat()
            })
            logger.info("Saved Q&A to SQL database.")

    def get_all_qa(self):
        """Retrieve all Q&A records from the SQL database."""
        select_sql = text("SELECT id, question, answer, model, valid, timestamp FROM qa_records")
        with self.engine.connect() as conn:
            result = conn.execute(select_sql)
            return [dict(row) for row in result]



class NoSQLDatabase:
    """
    NoSQL database integration using MongoDB.
    """
    def __init__(self, uri: str, db_name: str = "ragdb"):
        """
        Initialize the NoSQL database connection.
        Args:
            uri (str): MongoDB URI.
            db_name (str): Database name.
        """
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db["qa_records"]
            logger.info(f"Connected to MongoDB at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def save_qa(self, question: str, answer: str, model: str, valid: bool):
        """Save a Q&A record to the NoSQL database."""
        from datetime import datetime
        if self.collection:
            self.collection.insert_one({
                'question': question,
                'answer': answer,
                'model': model,
                'valid': valid,
                'timestamp': datetime.utcnow().isoformat()
            })
            logger.info("Saved Q&A to MongoDB.")

    def get_all_qa(self):
        """Retrieve all Q&A records from the NoSQL database."""
        if self.collection:
            return list(self.collection.find({}, {'_id': 0}))
        return []
