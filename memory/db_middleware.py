import sqlite3
import json
import os

class DatabaseMiddleware:
    def __init__(self, db_path="tradingagents.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                decision TEXT,
                pnl_percent REAL,
                reflection_text TEXT,
                math_stats TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def insert_reflection(self, record):
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO reflections (ticker, decision, pnl_percent, reflection_text, math_stats)
            VALUES (?, ?, ?, ?, ?)
        ''', (record.get('ticker'), record.get('decision'), record.get('pnl_percent'), 
              record.get('reflection_text'), record.get('math_stats')))
        self.conn.commit()

    def get_reflections(self, ticker, limit=10):
        c = self.conn.cursor()
        c.execute('''
            SELECT ticker, decision, pnl_percent, reflection_text, math_stats 
            FROM reflections WHERE ticker = ? ORDER BY id DESC LIMIT ?
        ''', (ticker, limit))
        rows = c.fetchall()
        return [{"ticker": r[0], "decision": r[1], "pnl_percent": r[2], "reflection_text": r[3], "math_stats": r[4]} for r in rows]
