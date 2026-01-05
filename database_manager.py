"""
UEL AI System - Database Management Module
"""

import sqlite3
from config import config
from utils import get_logger


class DatabaseManager:
    """Enhanced database management"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database_path
        self.logger = get_logger(f"{__name__}.DatabaseManager")
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_db(self):
        """Initialize database with all required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Students table (this will primarily be for other data if local files manage profiles)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
        ''')
        
        # Courses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS courses (
            id TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
        ''')
        
        # Applications table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (course_id) REFERENCES courses (id)
        )
        ''')
        
        # Conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            student_id TEXT,
            data TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
        ''')
        
        # Analytics table for tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_applications_student ON applications (student_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_applications_course ON applications (course_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_student ON conversations (student_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics (event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics (timestamp)')
        
        conn.commit()
        conn.close()