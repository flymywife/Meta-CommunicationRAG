# rag_models/model_vector_rerank.py

import numpy as np
from rag_models.base_model import BaseRAGModel
from typing import Any, Dict, Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import constants as c

class VectorSearchRerankModel(BaseRAGModel):
    def __init__(self, api_key, db_file='conversation_data.db'):
        try:
            super().__init__(api_key, db_file)

            # mMiniLM モデルとトークナイザーのロード
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(c.RERANK_MODEL)
                self.model = AutoModelForSequenceClassification.from_pretrained(c.RERANK_MODEL)
                self.model.eval()  # 推論モードに設定
            except Exception as e:
                raise Exception(f"リランカーモデルの読み込み中にエラーが発生しました: {str(e)}")

            # GPU が利用可能ならモデルを GPU に移動
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        except Exception as e:
            raise Exception(f"モデルの初期化中にエラーが発生しました: {str(e)}")

    def retrieve_context(self, query: str, task_name: str) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストを取得し、指定された形式の辞書を返します。
        """
        try:
            # ベクトルとコンテンツ、talk_numの取得
            vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)

            # クエリのベクトル化
            query_vector = self.vector_db.compute_query_embedding(query)

            # コサイン類似度の計算
            similarities = [self.vector_db.cosine_similarity(vector, query_vector) for vector in vectors]

            # 類似度の高い順にソートして上位K件を取得
            top_k = 5
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            top_contents = [contents[idx] for idx in sorted_indices]
            top_talk_nums = [talk_nums[idx] for idx in sorted_indices]

            # mMiniLM でリランキング
            reranked_contents, reranked_talk_nums = self.rerank_with_mMiniLM(query, top_contents, top_talk_nums)

            # get_contextを設定
            get_context = reranked_contents[0] if len(reranked_contents) > 0 else ''

            # get_talk_nums をカンマ区切りの文字列として設定
            get_talk_nums = ','.join(reranked_talk_nums[:2])

            # 結果を辞書で返す
            result = {
                'get_context': get_context,
                'get_talk_nums': get_talk_nums
            }

            return result

        except Exception as e:
            raise Exception(f"コンテキスト取得中にエラーが発生しました: {str(e)}")

    def rerank_with_mMiniLM(self, query: str, contents: List[str], talk_nums: List[str]) -> Tuple[List[str], List[str]]:
        """
        mMiniLM を使用してコンテンツをリランキングします。
        """
        try:
            # クエリと各コンテンツのペアを作成
            query_content_pairs = [(query, content) for content in contents]

            # トークン化とバッチ処理の準備
            inputs = self.tokenizer(
                [pair[0] for pair in query_content_pairs],
                [pair[1] for pair in query_content_pairs],
                return_tensors='pt',
                padding=True,
                truncation='only_second', 
                max_length=512
            )

            # デバイスに移動
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推論
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()  # スコアを取得

            # スコアに基づいてコンテンツをリランキング
            scored_items = list(zip(contents, talk_nums, scores))
            reranked_items = sorted(scored_items, key=lambda x: x[2], reverse=True)
            reranked_contents = [item[0] for item in reranked_items]
            reranked_talk_nums = [item[1] for item in reranked_items]

            return reranked_contents, reranked_talk_nums

        except Exception as e:
            raise Exception(f"リランキング処理中にエラーが発生しました: {str(e)}")