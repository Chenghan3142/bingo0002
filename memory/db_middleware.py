import sqlite3
import json
import os

class DatabaseMiddleware:
    def __init__(self, db_path="tradingagents.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        self._ensure_reward_columns()

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                decision TEXT,
                pnl_percent REAL,
                reward_score REAL,
                reward_text TEXT,
                reward_stats TEXT,
                reflection_text TEXT,
                math_stats TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def _ensure_reward_columns(self):
        c = self.conn.cursor()
        c.execute("PRAGMA table_info(reflections)")
        existing_columns = {row[1] for row in c.fetchall()}

        required_columns = {
            "reward_score": "REAL",
            "reward_text": "TEXT",
            "reward_stats": "TEXT",
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                c.execute(f"ALTER TABLE reflections ADD COLUMN {column_name} {column_type}")
        self.conn.commit()

    def insert_reflection(self, record):
        c = self.conn.cursor()
        # Ensure complex fields are serialized to JSON strings for SQLite
        reward_stats = record.get('reward_stats')
        math_stats = record.get('math_stats')
        try:
            if isinstance(reward_stats, (dict, list)):
                reward_stats = json.dumps(reward_stats, ensure_ascii=False)
        except Exception:
            reward_stats = str(reward_stats)

        try:
            if isinstance(math_stats, (dict, list)):
                math_stats = json.dumps(math_stats, ensure_ascii=False)
        except Exception:
            math_stats = str(math_stats)

        c.execute('''
            INSERT INTO reflections (
                ticker, decision, pnl_percent, reward_score, reward_text, reward_stats, reflection_text, math_stats
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.get('ticker'),
            record.get('decision'),
            record.get('pnl_percent'),
            record.get('reward_score'),
            record.get('reward_text'),
            reward_stats,
            record.get('reflection_text'),
            math_stats,
        ))
        self.conn.commit()

    def get_reflections(self, ticker, limit=10):
        c = self.conn.cursor()
        c.execute('''
            SELECT ticker, decision, pnl_percent, reward_score, reward_text, reward_stats, reflection_text, math_stats 
            FROM reflections WHERE ticker = ? ORDER BY id DESC LIMIT ?
        ''', (ticker, limit))
        rows = c.fetchall()
        result = []
        for r in rows:
            # attempt to deserialize JSON fields back to Python objects
            reward_stats = r[5]
            math_stats = r[7]
            try:
                if reward_stats is not None:
                    reward_stats = json.loads(reward_stats)
            except Exception:
                pass
            try:
                if math_stats is not None:
                    math_stats = json.loads(math_stats)
            except Exception:
                pass

            result.append({
                "ticker": r[0],
                "decision": r[1],
                "pnl_percent": r[2],
                "reward_score": r[3],
                "reward_text": r[4],
                "reward_stats": reward_stats,
                "reflection_text": r[6],
                "math_stats": math_stats,
            })
        return result
