# rag_models/model_hybrid.py

import numpy as np
import openai
from rag_models.base_model import BaseRAGModel
from typing import List, Dict, Any
import time
import constants as c
from rank_bm25 import BM25Okapi
import re

class HybridSearchModel(BaseRAGModel):
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db', k: int = 60):
        super().__init__(api_key, db_file)
        self.k = k  # RRFの調整パラメータ
        self.documents = []            # ドキュメントのテキスト
        self.document_ids = []         # ドキュメントのID（vector_id）
        self.talk_nums = []            # ドキュメントに対応する talk_num
        self.tokenized_documents = []  # BM25用のトークン化されたドキュメント
        self.bm25 = None               # BM25モデル
        self.doc_id_to_index = {}      # ドキュメントIDからインデックスへのマッピング

    def preprocess_text(self, text: str) -> str:
        # 日本語のテキストを前処理します（記号の除去など）
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        # テキストを前処理
        text = self.preprocess_text(text)
        # N-グラム（例えば2文字）に分割
        N = 2
        tokens = [text[i:i+N] for i in range(len(text)-N+1)]
        return tokens

    def load_data(self, task_name: str):
        # VectorDatabaseからベクトルとコンテンツを取得
        vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)
        if len(contents) == 0:
            raise ValueError(f"タスク名 '{task_name}' に対応するデータが存在しません。")

        self.documents = contents
        self.document_ids = vector_ids
        self.talk_nums = talk_nums

        # ドキュメントIDからインデックスへのマッピングを作成
        self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.document_ids)}

        # ドキュメントをトークナイズ
        self.tokenized_documents = [self.tokenize(content) for content in self.documents]

        # BM25モデルの作成
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def retrieve_context(self, query: str, task_name: str, qa_id: int) -> Dict[str, Any]:
        start_time = time.time()

        # データが未ロードの場合はロードする
        if not self.documents:
            self.load_data(task_name)

        # クエリの埋め込みベクトルを計算
        query_vector = self.vector_db.compute_query_embedding(query)

        # --- ベクトル検索によるランキング ---
        # FAISSインデックスから類似したベクトルを取得
        top_k = 100  # 必要に応じて調整
        vector_results = self.vector_db.retrieve_similar_vectors(query_vector, top_k)
        # ベクトル検索の順位辞書と類似度辞書を作成
        vector_rank_dict = {}
        vector_similarity_dict = {}
        for rank, (vector_id, similarity_score) in enumerate(vector_results):
            vector_rank_dict[vector_id] = rank
            vector_similarity_dict[vector_id] = similarity_score

        # --- BM25によるランキング ---
        # クエリをトークナイズ
        tokenized_query = self.tokenize(query)
        # BM25スコアを取得
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # スコアの降順でインデックスを取得
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        # ドキュメントIDと順位、スコアの辞書を作成
        bm25_rank_dict = {self.document_ids[idx]: rank for rank, idx in enumerate(bm25_ranking)}
        bm25_score_dict = {self.document_ids[idx]: bm25_scores[idx] for idx in bm25_ranking}

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

        # 選択されたドキュメントのコンテンツとスコアを取得
        selected_contents = []
        selected_talk_nums = []
        selected_similarities = []
        selected_bm25_scores = []
        selected_rrf_scores = []
        for doc_id in top_doc_ids:
            idx = self.doc_id_to_index[doc_id]
            content = self.documents[idx]
            talk_num = self.talk_nums[idx]
            selected_contents.append(content)
            selected_talk_nums.append(str(talk_num))
            # 類似度を取得（ベクトル検索の類似度を使用）
            similarity = vector_similarity_dict.get(doc_id, 0.0)
            selected_similarities.append(similarity)
            # BM25スコアを取得
            bm25_score = bm25_score_dict.get(doc_id, 0.0)
            selected_bm25_scores.append(bm25_score)
            # RRFスコアを取得
            rrf_score = rrf_scores.get(doc_id, 0.0)
            selected_rrf_scores.append(rrf_score)

        # コンテキストを設定
        get_context_1 = selected_contents[0] if len(selected_contents) > 0 else ''
        get_context_2 = selected_contents[1] if len(selected_contents) > 1 else ''
        get_context_3 = selected_contents[2] if len(selected_contents) > 2 else ''
        get_context_4 = selected_contents[3] if len(selected_contents) > 3 else ''
        get_context_5 = selected_contents[4] if len(selected_contents) > 4 else ''
        get_talk_nums = ','.join(selected_talk_nums)

        # 検索時間を計算
        processing_time = time.time() - start_time

        # task_id の取得
        task_id = self.database.get_task_id(task_name)

        # 結果をデータベースに保存
        result_entry = {
            'qa_id': qa_id,
            'task_name': task_name,
            'task_id': task_id,
            'talk_num_1': selected_talk_nums[0] if len(selected_talk_nums) > 0 else '',
            'talk_num_2': selected_talk_nums[1] if len(selected_talk_nums) > 1 else '',
            'talk_num_3': selected_talk_nums[2] if len(selected_talk_nums) > 2 else '',
            'talk_num_4': selected_talk_nums[3] if len(selected_talk_nums) > 3 else '',
            'talk_num_5': selected_talk_nums[4] if len(selected_talk_nums) > 4 else '',
            'cosine_similarity_1': selected_similarities[0] if len(selected_similarities) > 0 else 0.0,
            'cosine_similarity_2': selected_similarities[1] if len(selected_similarities) > 1 else 0.0,
            'cosine_similarity_3': selected_similarities[2] if len(selected_similarities) > 2 else 0.0,
            'cosine_similarity_4': selected_similarities[3] if len(selected_similarities) > 3 else 0.0,
            'cosine_similarity_5': selected_similarities[4] if len(selected_similarities) > 4 else 0.0,
            'BM25_score_1': selected_bm25_scores[0] if len(selected_bm25_scores) > 0 else 0.0,
            'BM25_score_2': selected_bm25_scores[1] if len(selected_bm25_scores) > 1 else 0.0,
            'BM25_score_3': selected_bm25_scores[2] if len(selected_bm25_scores) > 2 else 0.0,
            'BM25_score_4': selected_bm25_scores[3] if len(selected_bm25_scores) > 3 else 0.0,
            'BM25_score_5': selected_bm25_scores[4] if len(selected_bm25_scores) > 4 else 0.0,
            'rss_rank_1': selected_rrf_scores[0] if len(selected_rrf_scores) > 0 else 0.0,
            'rss_rank_2': selected_rrf_scores[1] if len(selected_rrf_scores) > 1 else 0.0,
            'rss_rank_3': selected_rrf_scores[2] if len(selected_rrf_scores) > 2 else 0.0,
            'rss_rank_4': selected_rrf_scores[3] if len(selected_rrf_scores) > 3 else 0.0,
            'rss_rank_5': selected_rrf_scores[4] if len(selected_rrf_scores) > 4 else 0.0,
            'processing_time': processing_time,
            'model': c.HYBRID_SEARCH,
            'created_at': self.database.get_current_timestamp()
        }

        # rag_results テーブルに挿入
        self.database.insert_rag_result(result_entry)

        result = {
            'get_context_1': get_context_1,
            'get_context_2': get_context_2,
            'get_context_3': get_context_3,
            'get_context_4': get_context_4,
            'get_context_5': get_context_5,
            'get_talk_nums': get_talk_nums
        }

        return result

    def close(self):
        super().close()
