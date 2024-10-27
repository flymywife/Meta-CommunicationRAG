# vector_db.py

import sqlite3
import openai
import numpy as np
import faiss  # FAISS をインポート
import os
from database import ConversationDatabase  # database.py をインポート

class VectorDatabase:
    def __init__(self, api_key, db_file='conversation_data.db', index_folder='faiss_index', index_file='faiss_index.index'):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file

        # インデックスフォルダのパスを設定
        self.index_folder = index_folder
        self.index_file = index_file  # インデックスファイル名

        # インデックスフォルダが存在しない場合は作成
        if not os.path.exists(self.index_folder):
            os.makedirs(self.index_folder)

        # インデックスファイルのフルパスを設定
        self.index_file_path = os.path.join(self.index_folder, self.index_file)

        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        self.create_vector_table()
        self.index = None  # FAISS インデックス
        self.ids = []      # ベクトルの ID リスト

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

        vectors = []
        ids = []

        for convo in conversations:
            content = convo['content']
            conversation_id = convo['conversation_id']
            try:
                # ベクトルの取得
                response = openai.Embedding.create(
                    input=content,
                    model="text-embedding-3-large"   # 適切なモデルを選択
                )
                vector = response['data'][0]['embedding']
                vector_np = np.array(vector, dtype='float32')
                vector_bytes = vector_np.tobytes()

                # ベクトルと関連情報をデータベースに保存
                insert_sql = '''
                INSERT INTO vector_table (conversation_id, task_id, content, vector)
                VALUES (?, ?, ?, ?)
                '''
                data = (
                    conversation_id,
                    convo['task_id'],
                    content,
                    vector_bytes
                )
                self.cursor.execute(insert_sql, data)
                self.conn.commit()

                # ベクトルと ID をリストに追加
                vectors.append(vector_np)
                ids.append(conversation_id)

            except Exception as e:
                print(f"ベクトル化中にエラーが発生しました (conversation_id: {conversation_id}): {e}")
                continue  # エラーが発生した場合は次の会話に進む

        # FAISS インデックスを作成してファイルに保存
        if vectors:
            self.create_faiss_index(np.array(vectors), ids)

        return len(conversations)  # ベクトル化した会話の数を返す

    def create_faiss_index(self, vectors, ids):
        dimension = vectors.shape[1]
        # インデックスの作成（ここではシンプルな L2 距離のフラットなインデックスを使用）
        index = faiss.IndexFlatL2(dimension)
        # ID を指定するために IndexIDMap を使用
        index = faiss.IndexIDMap(index)
        # インデックスにベクトルを追加
        index.add_with_ids(vectors, np.array(ids))

        # インデックスをファイルに保存
        faiss.write_index(index, self.index_file_path)
        print(f"FAISS インデックスを '{self.index_file_path}' に保存しました。")

    def load_faiss_index(self):
        if os.path.exists(self.index_file_path):
            self.index = faiss.read_index(self.index_file_path)
            print(f"FAISS インデックスを '{self.index_file_path}' から読み込みました。")
        else:
            print(f"FAISS インデックスファイル '{self.index_file_path}' が存在しません。")
            self.index = None

    def search(self, query, top_k=5):
        # インデックスがロードされていない場合はロード
        if self.index is None:
            self.load_faiss_index()
            if self.index is None:
                print("FAISS インデックスがロードされていません。")
                return []

        # クエリをベクトル化
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-3-large"  # 適切なモデルを選択
        )
        query_vector = np.array(response['data'][0]['embedding'], dtype='float32').reshape(1, -1)

        # インデックスから検索
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # マッチがない場合
            # データベースから対応するレコードを取得
            select_sql = '''
            SELECT content FROM vector_table WHERE conversation_id = ?
            '''
            self.cursor.execute(select_sql, (idx,))
            row = self.cursor.fetchone()
            if row:
                results.append({'conversation_id': idx, 'content': row[0], 'distance': dist})
        return results

    def close(self):
        self.conn.close()

def vectorize_and_store(api_key, task_name):
    vector_db = VectorDatabase(api_key=api_key)
    try:
        added_count = vector_db.vectorize_conversations(task_name=task_name)
        return added_count
    finally:
        vector_db.close()
