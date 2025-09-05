import sqlite3
from sqlite3 import Connection
import pandas as pd

DB_PATH = "data/emails.db"

def get_connection() -> Connection:
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            sender_name TEXT,
            subject TEXT,
            body TEXT,
            sent_date TEXT,
            priority TEXT,
            sentiment TEXT,
            phone TEXT,
            alt_email TEXT,
            requirement TEXT,
            preview TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_emails(df: pd.DataFrame):
    conn = get_connection()
    df.to_sql("emails", conn, if_exists="replace", index=False)
    conn.close()

def load_emails() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM emails", conn)
    conn.close()
    return df
