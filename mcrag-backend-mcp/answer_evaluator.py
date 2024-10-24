# answer_evaluator.py

import time
import json
import logging
from call_gpt import GPTClient  # GPTClientクラスをインポート
from rag_models.model_kmeans import KMeansModel  # KMeansModelをインポート
from rag_models.model_vector import VectorSearchModel  # VectorSearchModelをインポート
from rag_models.model_hybrid import HybridSearchModel  # HybridSearchModelをインポート
from rag_models.model_kmeans_hybrid import KMeansHybridModel  # KMeansHybridModelをインポート
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
            'クラスタリング': KMeansModel(api_key=self.api_key),
            'ベクトル検索': VectorSearchModel(api_key=self.api_key),
            'ハイブリッド検索': HybridSearchModel(api_key=self.api_key),
            'クラスタリングハイブリッド検索': KMeansHybridModel(api_key=self.api_key)
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
                talk_nums = entry.get('talk_nums', '')
                word = entry.get('word', '')
                query = entry.get('query', '')
                expected_answer = entry.get('answer', '')
                task_id = entry.get('task_id')
                word_info_id = entry.get('word_info_id')
                print(f"クエリを処理中 ({idx+1}/{len(data_list)}): {query}")

                # RAGモデルからコンテキストを取得
                context = rag_model.retrieve_context(query, task_name)
                print(f"取得したコンテキスト:{context}")

                # GPTにクエリとコンテキストを渡して回答を生成
                response_text, token_count, processing_time = self.generate_response(query, context)

                # 回答の比較
                is_correct, evaluation_detail = self.compare_answers(query, expected_answer, response_text)

                # 結果の保存
                self.total_tokens += token_count
                self.total_processing_time += processing_time

                result_entry = {
                    'talk_nums': talk_nums,
                    'task_name': task_name,
                    'task_id': task_id,
                    'word_info_id': word_info_id,
                    'word': word,
                    'query': query,
                    'expected_answer': expected_answer,
                    'gpt_response': response_text,
                    'is_correct': int(is_correct),
                    'evaluation_detail': evaluation_detail,
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
            raise e
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

    def generate_response(self, query, context):
        try:
            start_time = time.time()

            # コンテキストとクエリをプロンプトに組み合わせる
            prompt = f"以下は関連する会話の記録です。\n{context}\n\n質問: {query}\n\n上記の会話の記録に基づいて、この質問に対する適切な回答を指定されたキャラクターの口調を反映して提供してください。"

            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
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

    def compare_answers(self, query, expected_answer, actual_answer):
        try:
            # 評価用のプロンプトを作成
            evaluation_prompt = f"""
あなたは厳密な教師です。以下の質問に対する学生の回答が、模範解答と比べて正しいかどうかを判断してください。

質問:
{query}

模範解答:
{expected_answer}

学生の回答:
{actual_answer}

学生の回答が質問に適切に答えているかどうか、模範解答と比較して判断してください。正しい場合は1、正しくない場合は0を返してください。また、理由を簡潔に述べてください。
"""

            # Function Calling の定義
            functions = [
                {
                    "name": "evaluate_answer",
                    "description": "学生の回答が質問に適切に答えているかを評価します。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_correct": {
                                "type": "integer",
                                "description": "1（正しい）または 0（正しくない）"
                            },
                            "evaluation_detail": {
                                "type": "string",
                                "description": "評価の理由を簡潔に述べてください。"
                            }
                        },
                        "required": ["is_correct", "evaluation_detail"],
                    }
                }
            ]

            # GPTに評価を依頼
            response = self.gpt_client.call_gpt_function(
                messages=[
                    {"role": "user", "content": evaluation_prompt},
                ],
                functions=functions,
                function_call={"name": "evaluate_answer"},
                max_tokens=150,
                temperature=0.0,  # 評価なので温度は低めに
            )

            if not response:
                return 0, "評価に失敗しました。"

            # Function Call の結果を解析
            function_call = response['choices'][0]['message'].get('function_call')
            if function_call and function_call['name'] == 'evaluate_answer':
                arguments = json.loads(function_call['arguments'])
                is_correct = arguments.get('is_correct', 0)
                evaluation_detail = arguments.get('evaluation_detail', '評価の詳細がありません。')
            else:
                is_correct = 0
                evaluation_detail = "Function calling に失敗しました。"

            return is_correct, evaluation_detail  # 評価結果と理由を返す

        except Exception as e:
            print(f"評価中にエラーが発生しました: {e}")
            return 0, f"評価中にエラーが発生しました: {e}"

    def close(self):
        self.db.close()
