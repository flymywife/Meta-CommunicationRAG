# rag_models/model_kmeans_hybrid.py

import sqlite3
import numpy as np
import openai
from sklearn.cluster import KMeans
from rag_models.base_model import BaseRAGModel
from typing import List, Dict
from rank_bm25 import BM25Okapi
from janome.tokenizer import Tokenizer  # 日本語のトークナイザー

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    2つのベクトル間のコサイン類似度を計算します。
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # ゼロノルム対策
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

class KMeansHybridModel(BaseRAGModel):
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db', k: int = None, rrf_k: int = 60):
        """
        KMeansクラスタリングとハイブリッド検索モデルの初期化。

        Args:
            api_key (str): OpenAI APIキー。
            db_file (str): データベースファイル名。
            k (int): クラスタ数（ワードの数）。Noneの場合、自動で取得。
            rrf_k (int): RRFの調整パラメータ。
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file
        self.k = k  # クラスタ数
        self.rrf_k = rrf_k  # RRFの調整パラメータ
        # データベース接続の確立
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        # データ格納用のリストの初期化
        self.documents = []            # ドキュメントのテキスト
        self.document_ids = []         # ドキュメントのID（vector_id）
        self.embeddings = []           # ドキュメントの埋め込みベクトル
        self.tokenized_documents = []  # BM25用のトークン化されたドキュメント
        self.bm25 = None               # BM25モデル
        self.labels = []               # クラスタラベル
        self.tokenizer = Tokenizer()   # Janomeのトークナイザーを初期化

    def tokenize(self, text: str) -> List[str]:
        """
        テキストを形態素解析してトークンのリストを取得します。

        Args:
            text (str): トークン化するテキスト。

        Returns:
            List[str]: トークンのリスト。
        """
        return [token.surface for token in self.tokenizer.tokenize(text)]

    def fetch_vectors_and_contents(self, task_name: str):
        """
        特定のタスクに対応するベクトルとコンテンツをデータベースから取得します。

        Args:
            task_name (str): タスク名。
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
        if not rows:
            raise ValueError(f"タスク名 '{task_name}' に対応するデータが存在しません。")

        for row in rows:
            vector_id, content, vector_blob = row
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            self.document_ids.append(vector_id)
            self.documents.append(content)
            self.embeddings.append(vector)
            tokens = self.tokenize(content)
            self.tokenized_documents.append(tokens)

        self.embeddings = np.array(self.embeddings)

    def get_number_of_words(self, task_name: str) -> int:
        """
        words_info テーブルから特定のタスクに対応するワードの数を取得します。

        Args:
            task_name (str): タスク名。

        Returns:
            int: ワードの数。
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

    def cluster_vectors(self):
        """
        ベクトルを k 個のクラスタにクラスタリングします。
        """
        if len(self.embeddings) == 0:
            raise ValueError("クラスタリングするベクトルが存在しません。")

        if self.k is None:
            raise ValueError("クラスタ数が設定されていません。")

        if self.k <= 0:
            raise ValueError(f"クラスタ数が無効です: {self.k}")

        if len(self.embeddings) < self.k:
            self.k = len(self.embeddings)  # ベクトル数がクラスタ数より少ない場合の対処

        kmeans = KMeans(n_clusters=self.k, random_state=42)
        kmeans.fit(self.embeddings)
        self.labels = kmeans.labels_

    def compute_query_embedding(self, query: str) -> np.ndarray:
        """
        クエリを埋め込みベクトルに変換します。

        Args:
            query (str): クエリ文字列。

        Returns:
            np.ndarray: クエリの埋め込みベクトル。
        """
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_vector = np.array(response['data'][0]['embedding'], dtype=np.float32)
        return query_vector

    def get_most_similar_cluster(self, query_vector: np.ndarray) -> int:
        """
        最もコサイン類似度の高いチャンクを含むクラスタを特定します。

        Args:
            query_vector (np.ndarray): クエリの埋め込みベクトル。

        Returns:
            int: 最も類似度の高いクラスタのラベル。
        """
        # コサイン類似度の計算
        similarities = [cosine_similarity(vector, query_vector) for vector in self.embeddings]

        # クラスタごとに最大類似度を計算
        cluster_info = {}
        for idx, (label, sim) in enumerate(zip(self.labels, similarities)):
            if label not in cluster_info or sim > cluster_info[label]:
                cluster_info[label] = sim

        # 最大類似度が最も高いクラスタを特定
        best_cluster_label = max(cluster_info.items(), key=lambda x: x[1])[0]
        return best_cluster_label

    def retrieve_context(self, query: str, task_name: str) -> str:
        """
        クエリに関連するコンテキストを取得します。

        Args:
            query (str): ユーザーのクエリ。
            task_name (str): タスク名。

        Returns:
            str: 結合された関連コンテキスト。
        """
        # データのロード（未ロードの場合）
        if not self.documents:
            self.fetch_vectors_and_contents(task_name)
            # クラスタ数の取得（未設定の場合）
            if self.k is None:
                self.k = self.get_number_of_words(task_name)
            # ベクトルのクラスタリング
            self.cluster_vectors()
            # BM25モデルの作成
            self.bm25 = BM25Okapi(self.tokenized_documents)

        # クエリの埋め込みベクトルを計算
        query_vector = self.compute_query_embedding(query)

        # 最も類似度の高いクラスタを特定
        best_cluster_label = self.get_most_similar_cluster(query_vector)

        # クラスタ内のドキュメントを抽出
        cluster_indices = [idx for idx, label in enumerate(self.labels) if label == best_cluster_label]
        cluster_embeddings = [self.embeddings[idx] for idx in cluster_indices]
        cluster_documents = [self.documents[idx] for idx in cluster_indices]
        cluster_document_ids = [self.document_ids[idx] for idx in cluster_indices]
        cluster_tokenized_documents = [self.tokenized_documents[idx] for idx in cluster_indices]

        # --- ベクトル検索によるランキング（クラスタ内） ---
        similarities = [cosine_similarity(vector, query_vector) for vector in cluster_embeddings]
        vector_ranking = np.argsort(similarities)[::-1]
        vector_rank_dict = {cluster_document_ids[idx]: rank for rank, idx in enumerate(vector_ranking)}

        # --- BM25によるランキング（クラスタ内） ---
        tokenized_query = self.tokenize(query)
        bm25 = BM25Okapi(cluster_tokenized_documents)
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        bm25_rank_dict = {cluster_document_ids[idx]: rank for rank, idx in enumerate(bm25_ranking)}

        # --- RRFによるランキングの融合 ---
        rrf_scores = {}
        for doc_id in cluster_document_ids:
            rank_vec = vector_rank_dict.get(doc_id, None)
            rank_bm25 = bm25_rank_dict.get(doc_id, None)
            score = 0.0
            if rank_vec is not None:
                score += 1 / (self.rrf_k + rank_vec)
            if rank_bm25 is not None:
                score += 1 / (self.rrf_k + rank_bm25)
            rrf_scores[doc_id] = score

        # RRFスコアでドキュメントを降順ソート
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 上位N件を選択（必要に応じてNを調整）
        N = 5
        top_doc_ids = [doc_id for doc_id, _ in sorted_docs[:N]]
        selected_contents = [cluster_documents[cluster_document_ids.index(doc_id)] for doc_id in top_doc_ids]

        # 選択されたコンテンツを結合してコンテキストを生成
        context = "\n".join(selected_contents)
        return context

    def close(self):
        """
        データベース接続を閉じます。
        """
        self.conn.close()
