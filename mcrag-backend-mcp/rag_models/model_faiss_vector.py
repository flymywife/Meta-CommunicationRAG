# rag_models/faiss_rag_model.py

from .base_model import BaseRAGModel
from typing import Any, Dict
import numpy as np
import time
import constants as c


class FAISSVectorModel(BaseRAGModel):
    def retrieve_context(self, query: str, task_name: str, qa_id: int) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストをFAISSを使用して取得します。
        """
        start_time = time.time()

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
        selected_contents = []
        selected_talk_nums = []
        selected_similarities = []
        for vector_id, similarity in similar_vectors:
            if vector_id in vector_id_to_content:
                print(f"vector_id {vector_id} のコンテンツを取得中...")
                content = vector_id_to_content[vector_id]
                talk_num = vector_id_to_talk_num[vector_id]
                selected_contents.append(content)
                selected_talk_nums.append(str(talk_num))
                selected_similarities.append(float(similarity))  # numpy.float32 を float に変換
                print(f"取得成功: コンテンツ先頭100文字: {content[:100]}")
            else:
                print(f"vector_id {vector_id} は辞書に存在しません")

        # コンテキストを設定
        get_context_1 = selected_contents[0] if len(selected_contents) > 0 else ''
        get_context_2 = selected_contents[1] if len(selected_contents) > 0 else ''
        get_context_3 = selected_contents[2] if len(selected_contents) > 0 else ''
        get_context_4 = selected_contents[3] if len(selected_contents) > 0 else ''
        get_context_5 = selected_contents[4] if len(selected_contents) > 0 else ''
        get_talk_nums = ','.join(str(num) for num in selected_talk_nums)

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
            'model': c.VECTOR_SEARCH,
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
