# rag_models/model_vector.py

import sqlite3
import numpy as np
import openai
from rag_models.base_model import BaseRAGModel
from typing import List

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # ゼロノルム対策
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

class VectorSearchModel(BaseRAGModel):
    def __init__(self, api_key, db_file='conversation_data.db'):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()

    def fetch_vectors_and_contents(self, task_name):
        """
        特定のタスクに対応するベクトルとコンテンツをデータベースから取得します。
        """
        select_sql = '''
        SELECT v.vector_id, v.content, v.vector, c.word_info_id
        FROM vector_table v
        INNER JOIN conversations c ON v.conversation_id = c.conversation_id
        INNER JOIN tasks t ON c.task_id = t.task_id
        WHERE t.task_name = ?
        '''
        self.cursor.execute(select_sql, (task_name,))
        rows = self.cursor.fetchall()
        if not rows:
            raise ValueError(f"タスク名 '{task_name}' に対応するベクトルが存在しません。")

        vectors = []
        contents = []
        vector_ids = []
        word_info_ids = []
        for row in rows:
            vector_id, content, vector_blob, word_info_id = row
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            vectors.append(vector)
            contents.append(content)
            vector_ids.append(vector_id)
            word_info_ids.append(word_info_id)
        return np.array(vectors), contents, vector_ids, word_info_ids

    def get_conversation_count_per_word(self, task_name):
        """
        各ワードに対する会話履歴の総数を取得します。
        """
        select_sql = '''
        SELECT wi.word_info_id, COUNT(c.talk_num) AS num_conversations
        FROM conversations c
        INNER JOIN tasks t ON c.task_id = t.task_id
        INNER JOIN words_info wi ON c.word_info_id = wi.word_info_id
        WHERE t.task_name = ?
        GROUP BY wi.word_info_id;
        '''
        self.cursor.execute(select_sql, (task_name,))
        results = self.cursor.fetchall()
        if not results:
            raise ValueError(f"タスク名 '{task_name}' に対応する会話履歴が存在しません。")
        conversation_counts = {row[0]: row[1] for row in results}  # {word_info_id: count}
        return conversation_counts

    def compute_query_embedding(self, query):
        """
        クエリをベクトル化します。
        """
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_vector = np.array(response['data'][0]['embedding'], dtype=np.float32)
        return query_vector

    def retrieve_context(self, query: str, task_name: str) -> str:
        """
        クエリに関連するコンテキストを取得します。
        """
        # ベクトルとコンテンツの取得
        vectors, contents, vector_ids, word_info_ids = self.fetch_vectors_and_contents(task_name)

        # クエリのベクトル化
        query_vector = self.compute_query_embedding(query)

        # コサイン類似度の計算
        similarities = [cosine_similarity(vector, query_vector) for vector in vectors]

        # 各ワードに対する会話履歴の総数を取得
        conversation_counts = self.get_conversation_count_per_word(task_name)
        print(f"各ワードに対する会話履歴の総数: {conversation_counts}")

        # ワードごとに、コサイン類似度の高い順にコンテンツを選択
        word_contents = {}
        for idx, (word_info_id, similarity, content) in enumerate(zip(word_info_ids, similarities, contents)):
            if word_info_id not in word_contents:
                word_contents[word_info_id] = []
            word_contents[word_info_id].append((similarity, content))

        selected_contents = []
        for word_info_id, content_list in word_contents.items():
            # 類似度の高い順にソート
            sorted_contents = sorted(content_list, key=lambda x: x[0], reverse=True)
            # そのワードに対する会話数を取得
            N = conversation_counts.get(word_info_id, 0)
            # 上位N件を選択
            top_contents = [content for _, content in sorted_contents[:N]]
            selected_contents.extend(top_contents)

        # コンテキストとして結合
        context = "\n".join(selected_contents)
        return context

    def close(self):
        """
        データベース接続を閉じます。
        """
        self.conn.close()
