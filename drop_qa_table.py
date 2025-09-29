"""
drop_qa_table.py
Safely drops the qa_records table in rag.db so the RAG app can recreate it with the correct schema.
Usage: python drop_qa_table.py
"""

import sqlite3

def drop_qa_table(db_path='rag.db'):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('DROP TABLE IF EXISTS qa_records;')
        conn.commit()
        print("qa_records table dropped. Run your RAG app again to recreate it with the correct schema.")
    finally:
        conn.close()

if __name__ == "__main__":
    drop_qa_table()
