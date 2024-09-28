import sqlite3
import streamlit as st
from datetime import datetime

class ConversationDatabase:
    def __init__(self, db_file='conversation_data.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS conversations (
            talk_num INTEGER,
            task_name TEXT,
            word TEXT,
            user TEXT,
            assistant TEXT,
            token_count INTEGER,
            processing_time REAL,
            temperature REAL,
            created_at TEXT,
            PRIMARY KEY (talk_num, task_name, word)
        )
        '''
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def get_current_timestamp(self):
        # ミリ秒まで含めたタイムスタンプを取得
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def insert_conversation(self, entry):
        insert_sql = '''
        INSERT INTO conversations (talk_num, task_name, word, user, assistant, token_count, processing_time, temperature, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            entry['talk_num'],
            entry['task_name'],
            entry['word'],
            entry['user'],
            entry['assistant'],
            entry['token_count'],
            entry['processing_time'],
            entry['temperature'],
            entry['created_at']
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            st.error(f"データベースへの挿入中にエラーが発生しました: {e}")

    def close(self):
        self.conn.close()

    # 必要に応じて他の CRUD メソッドを追加できます
