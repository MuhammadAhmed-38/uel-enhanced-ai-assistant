import sqlite3
import json
from datetime import datetime

def create_minimal_database_schema(db_path: str):
    """Create minimal database schema for interview system"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Basic interview sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interview_sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        session_type TEXT NOT NULL,
        status TEXT NOT NULL,
        start_time TEXT,
        end_time TEXT,
        overall_score REAL,
        created_date TEXT
    )
    ''')
    
    # Basic responses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interview_responses (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        question_text TEXT NOT NULL,
        response_text TEXT NOT NULL,
        score REAL,
        timestamp TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Minimal database schema created!")

# Run the setup
create_minimal_database_schema('uel_ai_system.db')
