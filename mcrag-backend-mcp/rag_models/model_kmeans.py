# model_kmeans.py

import sqlite3
import numpy as np
import openai
from sklearn.cluster import KMeans
from rag_models.base_model import BaseRAGModel  # 抽象クラスをインポート
from typing import List

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # ゼロノルム対策
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

class KMeansModel(BaseRAGModel):
    def __init__(self, api_key, db_file='conversation_data.db'):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()

    def fetch_vectors_and_contents(self, task_name):
        """
        ベクトルと対応するコンテンツをデータベースから取得します（特定のタスクに限定）。
        """
        select_sql = '''
        SELECT v.vector_id, v.content, v.vector
        FROM vector_table v
        INNER JOIN conversations c ON v.conversation_id = c.conversation_id
        INNER JOIN tasks t ON c.task_id = t.task_id
        WHERE t.task_name = ?
        '''
        self.cursor.execute(select_sql, (task_name,))
        rows = self.cursor.fetchall()
        vectors = []
        contents = []
        vector_ids = []
        for row in rows:
            vector_id = row[0]
            content = row[1]
            vector_blob = row[2]
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            vectors.append(vector)
            contents.append(content)
            vector_ids.append(vector_id)
        return np.array(vectors), contents, vector_ids

    def get_number_of_words(self, task_name):
        """
        words_info テーブルから特定のタスクに対応するワードの数を取得します。
        """
        select_sql = '''
        SELECT COUNT(DISTINCT wi.word)
        FROM words_info wi
        INNER JOIN tasks t ON wi.task_id = t.task_id
        WHERE t.task_name = ?
        '''
        self.cursor.execute(select_sql, (task_name,))
        result = self.cursor.fetchone()
        if result and result[0] > 0:
            return result[0]
        else:
            raise ValueError(f"タスク名 '{task_name}' に対応するワードが存在しません。")

    def cluster_vectors(self, vectors, k):
        """
        ベクトルを k 個のクラスタにクラスタリングします。
        """
        if len(vectors) == 0:
            raise ValueError("クラスタリングするベクトルが存在しません。")

        if k <= 0:
            raise ValueError(f"クラスタ数が無効です: {k}")

        if len(vectors) < k:
            k = len(vectors)  # ベクトル数がクラスタ数より少ない場合の対処

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vectors)
        labels = kmeans.labels_
        return labels

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

    def get_most_similar_cluster(self, vectors, labels, query_vector, contents):
        """
        最もコサイン類似度の高いチャンクを含むクラスタを特定し、各クラスタの情報を表示します。
        """
        # コサイン類似度の計算
        similarities = [cosine_similarity(vector, query_vector) for vector in vectors]

        # クラスタごとに情報を収集
        cluster_info = {}
        for idx, (label, sim, content) in enumerate(zip(labels, similarities, contents)):
            if label not in cluster_info:
                cluster_info[label] = {
                    'max_similarity': sim,
                    'contents': []
                }
            else:
                if sim > cluster_info[label]['max_similarity']:
                    cluster_info[label]['max_similarity'] = sim
            cluster_info[label]['contents'].append({'similarity': sim, 'content': content})

        # クラスタごとの最大類似度でソート
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['max_similarity'], reverse=True)

        # 各クラスタの情報を表示
        print("各クラスタの情報（最大類似度の高い順）:")
        for label, info in sorted_clusters:
            print(f"\nクラスタ: {label}, 最大類似度: {info['max_similarity']:.4f}")
            # 各コンテンツを類似度の高い順にソート
            sorted_contents = sorted(info['contents'], key=lambda x: x['similarity'], reverse=True)
            for item in sorted_contents:
                print(f"  類似度: {item['similarity']:.4f}, コンテンツ: {item['content']}")

        # 最大類似度が最も高いクラスタを返す
        best_cluster_label = sorted_clusters[0][0]
        return best_cluster_label, similarities

    def retrieve_contents_from_cluster(self, labels, contents, vector_ids, best_cluster_label):
        """
        最も類似度の高いクラスタに属するコンテンツを取得します。
        """
        selected_contents = []
        selected_vector_ids = []
        for label, content, vector_id in zip(labels, contents, vector_ids):
            if label == best_cluster_label:
                selected_contents.append(content)
                selected_vector_ids.append(vector_id)
        return selected_contents, selected_vector_ids

    def retrieve_context(self, query: str, task_name: str) -> str:
        """
        会話履歴からクエリに関連するコンテキストを取得する
        """
        # ベクトルとコンテンツの取得（特定のタスクに限定）
        vectors, contents, vector_ids = self.fetch_vectors_and_contents(task_name)

        if len(vectors) == 0:
            raise ValueError(f"タスク名 '{task_name}' に対応するベクトルが存在しません。ベクトル化が実行されているか確認してください。")

        # クラスタ数（ワードの数）の取得
        k = self.get_number_of_words(task_name)

        if k <= 0:
            raise ValueError(f"タスク名 '{task_name}' に対応するワードの数がゼロまたは負の値です。")

        # ベクトルのクラスタリング
        labels = self.cluster_vectors(vectors, k)

        # クエリのベクトル化
        query_vector = self.compute_query_embedding(query)

        # 最も類似度の高いクラスタを特定し、クラスタ情報を表示
        best_cluster_label, similarities = self.get_most_similar_cluster(vectors, labels, query_vector, contents)

        # クラスタからコンテンツを取得
        selected_contents, selected_vector_ids = self.retrieve_contents_from_cluster(labels, contents, vector_ids, best_cluster_label)

        # コンテキストとして、選択されたコンテンツを結合して返す
        context = "\n".join(selected_contents)
        return context

    def close(self):
        """
        データベース接続を閉じます。
        """
        self.conn.close()
