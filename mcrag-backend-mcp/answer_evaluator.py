# answer_evaluator.py

import time
import json
import logging
from call_gpt import GPTClient  # GPTClientクラスをインポート
from rag_models.model_vector_rerank import VectorSearchRerankModel  # VectorSearchRerankModelをインポート
from rag_models.model_vector import VectorSearchModel  # VectorSearchModelをインポート
from rag_models.model_faiss_rag import FAISSRAGModel  # VectorSearchModelをインポート
from database import ConversationDatabase, DataAlreadyExistsError, DataNotFoundError  # データベースクラスをインポート

class AnswerEvaluator:
    def __init__(self, api_key, temperature=0.7, db_file='conversation_data.db'):
        self.temperature = temperature
        self.api_key = api_key
        self.total_tokens = 0
        self.total_processing_time = 0

        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

        # RAGモデルのインスタンスを作成
        self.rag_models = {
            'ベクトル検索': VectorSearchModel(api_key=self.api_key),
            'FAISS検索': FAISSRAGModel(api_key=self.api_key)
        }

        # データベースのインスタンスを作成
        self.db = ConversationDatabase(db_file=db_file)

    def evaluate_answers(self, task_name):
        # データベースから質問と回答を取得
        data_list = self.get_qas_from_db(task_name)

        if not data_list:
            raise DataNotFoundError(f"タスク名 '{task_name}' に対応する質問と回答が見つかりません。")

        results = []

        for model_name, rag_model in self.rag_models.items():
            print(f"\nモデル '{model_name}' の評価を開始します。\n")
            # 評価結果の重複チェック（モデルごとにチェック）
            if self.db.has_evaluated_answers(task_name, model_name):
                print(f"タスク名 '{task_name}' の評価結果は既に存在します。モデル: {model_name}")
                continue  # 既に評価済みの場合は次のモデルへ

            for idx, entry in enumerate(data_list):
                qa_id = entry.get('qa_id', '')
                talk_nums = entry.get('talk_nums', '')
                word = entry.get('word', '')
                question = entry.get('question', '')
                expected_answer = entry.get('answer', '')
                task_id = entry.get('task_id')
                word_info_id = entry.get('word_info_id')
                print(f"クエリを処理中 ({idx+1}/{len(data_list)}): {question}")

                # RAGモデルからコンテキストを取得
                context_result = rag_model.retrieve_context(question, task_name, qa_id)
                print(f"取得したコンテキスト: {context_result}")

                # コンテキストの内容を取得
                get_context = context_result.get('get_context', '')
                get_talk_nums = context_result.get('get_talk_nums', '')

                # GPTにクエリとコンテキストを渡して回答を生成
                response_text, token_count, processing_time = self.generate_response(question, get_context)

                # 結果の保存
                self.total_tokens += token_count
                self.total_processing_time += processing_time

                result_entry = {
                    'qa_id': qa_id,
                    'talk_nums': talk_nums,
                    'task_name': task_name,
                    'task_id': task_id,
                    'word_info_id': word_info_id,
                    'word': word,
                    'question': question,
                    'expected_answer': expected_answer,
                    'gpt_response': response_text,
                    'get_context': get_context,
                    'get_talk_nums': get_talk_nums,
                    'token_count': token_count,
                    'processing_time': processing_time,
                    'model': model_name  # モデル名を追加
                }
                results.append(result_entry)

                # 評価結果をデータベースに保存
                self.save_evaluated_answer(result_entry)

        return results

    def save_evaluated_answer(self, result_entry):
        try:
            self.db.insert_evaluated_answer(result_entry)
        except DataAlreadyExistsError as e:
            print(f"評価結果は既に存在します。詳細: {e}")
        except Exception as e:
            logging.error(f"評価結果の保存中にエラーが発生しました: {e}")
            raise e

    def get_qas_from_db(self, task_name):
        # データベースから質問と回答を取得
        try:
            qas_list = self.db.get_generated_qas_with_ids_by_task_name(task_name)
            return qas_list
        except Exception as e:
            print(f"データベースからのデータ取得中にエラーが発生しました: {e}")
            return []

    def generate_response(self, question, context):
        try:
            start_time = time.time()

            # コンテキストとクエリをプロンプトに組み合わせる
            context_text = f"{context}".strip()
            prompt = f"以下は関連する会話の記録です。\n{context_text}\n\n質問: {question}\n\n上記の会話の記録に基づいて、この質問に対する適切な回答を指定されたキャラクターの口調を反映して提供してください。"

            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0

            end_time = time.time()
            processing_time = end_time - start_time

            message = response['choices'][0]['message']['content'].strip()
            token_count = response['usage']['total_tokens']

            return message, token_count, processing_time
        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {e}")
            return "", 0, 0

    def close(self):
        self.db.close()
