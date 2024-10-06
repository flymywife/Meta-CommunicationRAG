# database.py

import sqlite3
import streamlit as st
from datetime import datetime

class ConversationDatabase:
    def __init__(self, db_file='conversation_data.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")  # 外部キー制約を有効化
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # tasks テーブル
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL UNIQUE,
            character_prompt TEXT,
            user_prompt TEXT
        )
        ''')
        # words_info テーブル
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS words_info (
            word_info_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            word TEXT,
            info TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        )
        ''')
        # conversations テーブル
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            word_info_id INTEGER NOT NULL,
            talk_num TEXT NOT NULL,
            user TEXT,
            assistant TEXT,
            token_count TEXT,
            processing_time TEXT,
            temperature TEXT,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (word_info_id) REFERENCES words_info (word_info_id),
            UNIQUE (task_id, word_info_id, talk_num)
        )
        ''')
        self.conn.commit()

    def get_current_timestamp(self):
        # ミリ秒まで含めたタイムスタンプを取得
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def insert_task(self, task):
        try:
            insert_sql = '''
            INSERT INTO tasks (task_name, character_prompt, user_prompt)
            VALUES (?, ?, ?)
            '''
            data = (
                task['task_name'],
                task.get('character_prompt', ''),
                task.get('user_prompt', '')
            )
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
            return self.cursor.lastrowid  # 挿入したタスクのIDを返す
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed: tasks.task_name' in str(e):
                st.error(f"タスク名 '{task['task_name']}' は既に存在します。別のタスク名を使用してください。")
            else:
                st.error(f"タスクの挿入中にエラーが発生しました: {e}")
            return None
        except Exception as e:
            st.error(f"タスクの挿入中にエラーが発生しました: {e}")
            return None

    def insert_word_info(self, task_id, word, info):
        try:
            insert_sql = '''
            INSERT INTO words_info (task_id, word, info)
            VALUES (?, ?, ?)
            '''
            data = (task_id, word, info)
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
            return self.cursor.lastrowid  # 挿入したワード情報のIDを返す
        except Exception as e:
            st.error(f"ワード情報の挿入中にエラーが発生しました: {e}")
            return None

    def insert_conversation(self, entry):
        insert_sql = '''
        INSERT INTO conversations (task_id, word_info_id, talk_num, user, assistant, token_count, processing_time, temperature, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            entry['task_id'],
            entry['word_info_id'],
            entry['talk_num'],
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
        except Exception as e:
            st.error(f"データベースへの挿入中にエラーが発生しました: {e}")

    def close(self):
        self.conn.close()
