# rag_models/model_hybrid.py

import sqlite3
import numpy as np
import openai
from rag_models.base_model import BaseRAGModel
from typing import List, Dict
from rank_bm25 import BM25Okapi

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    2つのベクトル間のコサイン類似度を計算します。
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # ゼロノルム対策（ノルムが0の場合は類似度を0とする）
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    # コサイン類似度の計算
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

class HybridSearchModel(BaseRAGModel):
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db', k: int = 60):
        """
        ハイブリッド検索モデルの初期化。

        Args:
            api_key (str): OpenAI APIキー。
            db_file (str): データベースファイル名。
            k (int): RRFの調整パラメータ。
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file
        self.k = k  # RRFの調整パラメータ
        # データベース接続の確立
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        # データの格納用リストの初期化
        self.documents = []            # ドキュメントのテキスト
        self.document_ids = []         # ドキュメントのID（vector_id）
        self.embeddings = []           # ドキュメントの埋め込みベクトル
        self.tokenized_documents = []  # BM25用のトークン化されたドキュメント
        self.bm25 = None               # BM25モデル

    def load_data(self, task_name: str):
        """
        特定のタスクに対応するドキュメントと埋め込みベクトルをデータベースからロードします。

        Args:
            task_name (str): タスク名。
        """
        # データベースからドキュメントとベクトルを取得するSQLクエリ
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
            # ベクトルデータをNumPy配列に変換
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            self.document_ids.append(vector_id)
            self.documents.append(content)
            self.embeddings.append(vector)
            # BM25用にトークン化（日本語の場合は形態素解析を推奨）
            self.tokenized_documents.append(content.split())

        # 埋め込みベクトルをNumPy配列に変換
        self.embeddings = np.array(self.embeddings)
        # BM25モデルの作成
        self.bm25 = BM25Okapi(self.tokenized_documents)

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

    def retrieve_context(self, query: str, task_name: str) -> str:
        """
        クエリに関連するコンテキストを取得します。

        Args:
            query (str): ユーザーのクエリ。
            task_name (str): タスク名。

        Returns:
            str: 結合された関連コンテキスト。
        """
        # データが未ロードの場合はロードする
        if not self.documents:
            self.load_data(task_name)

        # クエリの埋め込みベクトルを計算
        query_vector = self.compute_query_embedding(query)

        # --- ベクトル検索によるランキング ---
        # 各ドキュメントとのコサイン類似度を計算
        similarities = [cosine_similarity(vector, query_vector) for vector in self.embeddings]
        # 類似度の降順でインデックスを取得
        vector_ranking = np.argsort(similarities)[::-1]
        # ドキュメントIDと順位の辞書を作成
        vector_rank_dict = {self.document_ids[idx]: rank for rank, idx in enumerate(vector_ranking)}

        # --- BM25によるランキング ---
        # クエリをトークン化
        tokenized_query = query.split()
        # BM25スコアを取得
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # スコアの降順でインデックスを取得
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        # ドキュメントIDと順位の辞書を作成
        bm25_rank_dict = {self.document_ids[idx]: rank for rank, idx in enumerate(bm25_ranking)}

        # --- RRFによるランキングの融合 ---
        # 各ドキュメントIDに対してRRFスコアを計算
        rrf_scores = {}
        for doc_id in self.document_ids:
            # ベクトル検索の順位を取得
            rank_vec = vector_rank_dict.get(doc_id, None)
            # BM25の順位を取得
            rank_bm25 = bm25_rank_dict.get(doc_id, None)
            # RRFスコアの初期化
            score = 0.0
            # ベクトル検索の順位が存在する場合、RRFスコアに加算
            if rank_vec is not None:
                score += 1 / (self.k + rank_vec)
            # BM25の順位が存在する場合、RRFスコアに加算
            if rank_bm25 is not None:
                score += 1 / (self.k + rank_bm25)
            # ドキュメントIDとスコアを記録
            rrf_scores[doc_id] = score

        # RRFスコアでドキュメントを降順ソート
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 上位N件を選択（必要に応じてNを調整）
        N = 5
        top_doc_ids = [doc_id for doc_id, _ in sorted_docs[:N]]
        # 選択されたドキュメントのコンテンツを取得
        selected_contents = [self.documents[self.document_ids.index(doc_id)] for doc_id in top_doc_ids]

        # 選択されたコンテンツを結合してコンテキストを生成
        context = "\n".join(selected_contents)
        return context

    def close(self):
        """
        データベース接続を閉じます。
        """
        self.conn.close()
