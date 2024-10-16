# vector_db.py

import sqlite3
import openai
import numpy as np
from database import ConversationDatabase  # database.py をインポート

class VectorDatabase:
    def __init__(self, api_key, db_file='conversation_data.db'):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        self.create_vector_table()

    def create_vector_table(self):
        # vector_table テーブルの作成
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_table (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            task_id INTEGER NOT NULL,
            content TEXT,
            vector BLOB,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        )
        ''')
        self.conn.commit()

    def vectorize_conversations(self, task_name):
        # ConversationDatabase を使用して会話データを取得
        convo_db = ConversationDatabase(db_file=self.db_file)
        conversations = convo_db.get_conversations_by_task_name(task_name)
        convo_db.close()

        if not conversations:
            raise ValueError(f"タスク名 '{task_name}' に一致する会話が見つかりません。")

        for convo in conversations:
            content = convo['content']
            try:
                # ベクトルの取得
                response = openai.Embedding.create(
                    input=content,
                    model="text-embedding-ada-002"
                )
                vector = response['data'][0]['embedding']
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()

                # ベクトルと関連情報をデータベースに保存
                insert_sql = '''
                INSERT INTO vector_table (conversation_id, task_id, content, vector)
                VALUES (?, ?, ?, ?)
                '''
                data = (
                    convo['conversation_id'],
                    convo['task_id'],
                    content,
                    vector_bytes
                )
                self.cursor.execute(insert_sql, data)
                self.conn.commit()
            except Exception as e:
                print(f"ベクトル化中にエラーが発生しました (conversation_id: {convo['conversation_id']}): {e}")
                continue  # エラーが発生した場合は次の会話に進む

        return len(conversations)  # ベクトル化した会話の数を返す

    def close(self):
        self.conn.close()

def vectorize_and_store(api_key, task_name):
    vector_db = VectorDatabase(api_key=api_key)
    try:
        added_count = vector_db.vectorize_conversations(task_name=task_name)
        return added_count
    finally:
        vector_db.close()
