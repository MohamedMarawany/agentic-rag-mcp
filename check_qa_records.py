import sqlite3
import os
db_path = 'rag.db'
print("DB absolute path:", os.path.abspath(db_path))
conn = sqlite3.connect(db_path)
rows = conn.execute('SELECT * FROM qa_records').fetchall()
print(rows)
conn.close()