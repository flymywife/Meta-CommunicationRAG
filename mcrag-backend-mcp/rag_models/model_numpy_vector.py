# rag_models/model_vector.py

import numpy as np
from rag_models.base_model import BaseRAGModel
from typing import Dict, Any
import time

class VectorSearchModel(BaseRAGModel):
    def __init__(self, api_key, db_file='conversation_data.db'):
        super().__init__(api_key, db_file)
        # BaseRAGModelでself.databaseとself.vector_dbが初期化されていることを想定

    def retrieve_context(self, query: str, task_name: str, qa_id: int) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストを取得し、指定された形式の辞書を返します。
        また、検索結果を rag_results テーブルに保存します。
        """
        start_time = time.time()
        
        # ベクトルとコンテンツ、talk_numの取得
        vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)

        # クエリのベクトル化
        query_vector = self.vector_db.compute_query_embedding(query)

        # コサイン類似度の計算
        similarities = [self.vector_db.cosine_similarity(vector, query_vector) for vector in vectors]

        # 類似度の高い順にソート
        sorted_indices = np.argsort(similarities)[::-1]

        # 上位N件を取得（例として上位5件）
        top_k = 5
        top_indices = sorted_indices[:top_k]
        selected_contents = [contents[idx] for idx in top_indices]
        selected_talk_nums = [talk_nums[idx] for idx in top_indices]
        selected_similarities = [similarities[idx] for idx in top_indices]

        # get_context
        get_context = selected_contents[0] if len(selected_contents) > 0 else ''

        # get_talk_nums をカンマ区切りで設定
        get_talk_nums = str(selected_talk_nums[0])

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
            'processing_time': processing_time,
            'model': 'numpyベクトル検索',
            'created_at': self.database.get_current_timestamp()
        }

        # rag_results テーブルに挿入
        self.database.insert_rag_result(result_entry)

        # 結果を辞書で返す
        result = {
            'get_context': get_context,
            'get_talk_nums': get_talk_nums
        }

        return result
