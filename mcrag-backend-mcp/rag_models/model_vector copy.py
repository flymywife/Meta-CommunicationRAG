# rag_models/model_vector.py

import numpy as np
from rag_models.base_model import BaseRAGModel
from typing import Dict, Any
import constants as c

class VectorSearchModel(BaseRAGModel):
    def __init__(self, api_key, db_file='conversation_data.db'):
        super().__init__(api_key, db_file)

    def retrieve_context(self, query: str, task_name: str) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストを取得し、指定された形式の辞書を返します。
        """
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

        # get_context_1 と get_context_2 を設定
        get_context_1 = selected_contents[0] if len(selected_contents) > 0 else ''
        get_context_2 = selected_contents[1] if len(selected_contents) > 1 else ''

        # get_talk_nums をカンマ区切りで設定
        get_talk_nums = ','.join(selected_talk_nums[:2])

        # 結果を辞書で返す
        result = {
            'get_context_1': get_context_1,
            'get_context_2': get_context_2,
            'get_talk_nums': get_talk_nums
        }

        return result
