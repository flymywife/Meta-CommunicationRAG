# rag_models/model_bm25.py

import numpy as np
from rag_models.base_model import BaseRAGModel
from typing import List, Dict, Any
import time
import constants as c
from rank_bm25 import BM25Okapi
import re

class BM25SearchModel(BaseRAGModel):
    """
    ベクトル検索を使わず、BM25 のみでコンテキストを取得する RAG 用クラス。
    """
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db'):
        super().__init__(api_key, db_file)
        self.documents = []            
        self.talk_nums = []            
        self.tokenized_documents = []  
        self.bm25 = None               

    def preprocess_text(self, text: str) -> str:
        """
        日本語テキストの前処理（不要な記号などの除去）。
        """
        text = re.sub(r'[^\wぁ-んァ-ン一-龥]', '', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        テキストを2文字N-gramでトークナイズする（BM25用）。
        """
        text = self.preprocess_text(text)
        N = 2
        tokens = [text[i:i+N] for i in range(len(text) - N + 1)]
        return tokens

    def load_data(self, task_name: str):
        """
        ベクトルDBからテキストを取得し、BM25を初期化する。
        本来はベクトル検索用に使っていたが、ここでは contents (documents) と talk_nums だけ使う。
        """
        # fetch_vectors_and_contents でドキュメント本文と talk_num を取得
        # vectors は使用しないが、content/talk_num はここで流用
        vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)
        if len(contents) == 0:
            raise ValueError(f"タスク名 '{task_name}' に対応するデータが存在しません。")

        self.documents = contents
        self.talk_nums = talk_nums

        # ドキュメントをトークナイズ
        self.tokenized_documents = [self.tokenize(doc) for doc in self.documents]

        # BM25モデルの作成
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def retrieve_context(self, query: str, task_name: str, qa_id: int) -> Dict[str, Any]:
        """
        BM25 のみを用いた検索で上位5件のコンテキストを返す。
        """
        start_time = time.time()

        # まだデータをロードしていなければロード
        if not self.documents:
            self.load_data(task_name)

        # クエリをトークナイズしてBM25スコアを計算
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # スコアの降順でソートし、上位 N 件を取得
        N = 5
        ranking_indices = np.argsort(bm25_scores)[::-1]
        top_indices = ranking_indices[:N]

        # 上位N件を取り出してリスト化
        selected_contents = []
        selected_talk_nums = []
        selected_bm25_scores = []
        for idx in top_indices:
            content = self.documents[idx]
            talk_num = str(self.talk_nums[idx])
            score = bm25_scores[idx]

            selected_contents.append(content)
            selected_talk_nums.append(talk_num)
            selected_bm25_scores.append(score)

        # 足りない場合は空文字で埋める
        while len(selected_contents) < N:
            selected_contents.append('')
            selected_talk_nums.append('')
            selected_bm25_scores.append(0.0)

        get_context_1 = selected_contents[0]
        get_context_2 = selected_contents[1]
        get_context_3 = selected_contents[2]
        get_context_4 = selected_contents[3]
        get_context_5 = selected_contents[4]
        get_talk_nums = ','.join(selected_talk_nums)

        processing_time = time.time() - start_time
        task_id = self.database.get_task_id(task_name)

        # DBに検索結果を保存
        result_entry = {
            'qa_id': qa_id,
            'task_name': task_name,
            'task_id': task_id,
            # 該当トーク履歴番号
            'talk_num_1': selected_talk_nums[0],
            'talk_num_2': selected_talk_nums[1],
            'talk_num_3': selected_talk_nums[2],
            'talk_num_4': selected_talk_nums[3],
            'talk_num_5': selected_talk_nums[4],
            # BM25スコア
            'BM25_score_1': selected_bm25_scores[0],
            'BM25_score_2': selected_bm25_scores[1],
            'BM25_score_3': selected_bm25_scores[2],
            'BM25_score_4': selected_bm25_scores[3],
            'BM25_score_5': selected_bm25_scores[4],
            # ベクトル検索の類似度は使わないため 0.0 を入れておく（必要に応じてカラム削除OK）
            'cosine_similarity_1': 0.0,
            'cosine_similarity_2': 0.0,
            'cosine_similarity_3': 0.0,
            'cosine_similarity_4': 0.0,
            'cosine_similarity_5': 0.0,
            # 処理時間など
            'processing_time': processing_time,
            'model': c.BM25_SEARCH,
            'created_at': self.database.get_current_timestamp()
        }
        self.database.insert_rag_result(result_entry)

        return {
            'get_context_1': get_context_1,
            'get_context_2': get_context_2,
            'get_context_3': get_context_3,
            'get_context_4': get_context_4,
            'get_context_5': get_context_5,
            'get_talk_nums': get_talk_nums
        }

    def close(self):
        super().close()
