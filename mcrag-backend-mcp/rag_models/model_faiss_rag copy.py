# rag_models/faiss_rag_model.py

from .base_model import BaseRAGModel
from typing import Any, Dict
import numpy as np

class FAISSRAGModel(BaseRAGModel):
    def retrieve_context(self, query: str, task_name: str) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストをFAISSを使用して取得します。
        """
        # データベースの状態を確認
        vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)
        print(f"データベースの現在の状態:")
        print(f"vector_ids in DB: {vector_ids}")
        print(f"contents数: {len(contents)}")
        print(f"talk_nums: {talk_nums}")

        # マッピングを作成
        vector_id_to_content = dict(zip(vector_ids, contents))
        vector_id_to_talk_num = dict(zip(vector_ids, talk_nums))

        # クエリのベクトル化と検索
        query_vector = self.vector_db.compute_query_embedding(query)
        similar_vectors = self.vector_db.retrieve_similar_vectors(query_vector, top_k=5)
        
        # 直接マッピングを使用してコンテンツを取得
        contents = []
        talk_nums = []
        for vector_id, similarity in similar_vectors:
            if vector_id in vector_id_to_content:
                print(f"vector_id {vector_id} のコンテンツを取得中...")
                content = vector_id_to_content[vector_id]
                talk_num = vector_id_to_talk_num[vector_id]
                contents.append(content)
                talk_nums.append(str(talk_num))
                print(f"取得成功: コンテンツ先頭100文字: {content[:100]}")
            else:
                print(f"vector_id {vector_id} は辞書に存在しません")

        # コンテキストを設定
        get_context_1 = contents[0] if len(contents) > 0 else ''
        get_context_2 = contents[1] if len(contents) > 1 else ''
        get_talk_nums = ','.join(talk_nums[:2])

        result = {
            'get_context_1': get_context_1,
            'get_context_2': get_context_2,
            'get_talk_nums': get_talk_nums
        }

        return result