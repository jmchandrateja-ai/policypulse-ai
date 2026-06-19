import sqlite3
import os

def setup_database():
    conn = sqlite3.connect("data/complaints.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_type TEXT,
            original_text TEXT,
            cleaned_text TEXT,
            domain TEXT,
            sentiment TEXT,
            stance TEXT,
            urgency TEXT,
            confidence REAL,
            routing TEXT,
            latitude REAL,
            longitude REAL,
            area TEXT,
            ward TEXT,
            media_path TEXT,
            is_public INTEGER DEFAULT 1,
            party_context TEXT,
            legislator_brief TEXT,
            confirmation_audio TEXT,
            status TEXT DEFAULT 'open'
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT,
            area TEXT,
            complaint_count INTEGER,
            urgency_score REAL,
            last_updated TEXT,
            latitude REAL,
            longitude REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database created successfully")

if __name__ == "__main__":
    setup_database()
