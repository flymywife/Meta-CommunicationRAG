# answer_evaluator.py
import time
import logging
from call_gpt import GPTClient  # GPTClientクラスをインポート
from rag_models.model_subject import SubjectSearchModel  # SubjectSearchModelをインポート
from rag_models.model_hybrid import HybridSearchModel  # HybridSearchModelをインポート
from rag_models.model_faiss_vector import FAISSVectorModel  # VectorSearchModelをインポート
from rag_models.model_topic_word import TopicWordSearchModel  # VectorSearchModelをインポート
from rag_models.model_bm25 import BM25SearchModel  # VectorSearchModelをインポート
from database import ConversationDatabase, DataAlreadyExistsError, DataNotFoundError  # データベースクラスをインポート
import constants as c



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
            c.VECTOR_SEARCH: FAISSVectorModel(api_key=self.api_key),
            c.HYBRID_SEARCH: HybridSearchModel(api_key=self.api_key),
            c.SUBJECT_SEARCH: SubjectSearchModel(api_key=self.api_key),
            c.TOPIC_WORD_SEARCH: TopicWordSearchModel(api_key=self.api_key),
            c.BM25_SEARCH: BM25SearchModel(api_key=self.api_key)
        }

        # データベースのインスタンスを作成
        self.db = ConversationDatabase(db_file=db_file)

    def evaluate_answers(self, task_name):
        # データベースから質問と回答を取得
        data_list = self.db.get_generated_qas_with_ids_by_task_name(task_name)

        if not data_list:
            raise DataNotFoundError(f"タスク名 '{task_name}' に対応する質問と回答が見つかりません。")

        # タスクプロンプトを取得
        character_prompt = self.db.get_tasks_character_prompt(task_name)

        # data_listにプロンプトを追加
        for entry in data_list:
            entry.update({'character_prompt': character_prompt})

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
                get_contexts = '\n'.join([
                    context_result.get(f'get_context_{i}', '') 
                    for i in range(1, 6)
                ])
                get_talk_nums = context_result.get('get_talk_nums', '')

                # GPTにクエリとコンテキストを渡して回答を生成
                # token_count の代わりに input_token, output_token を使用するため、prompt_tokens, completion_tokensを取得する
                response_text, total_tokens, prompt_tokens, completion_tokens, processing_time = self.generate_response(
                    question, get_contexts, character_prompt, get_talk_nums, task_name
                )

                # input_token, output_token を計算
                input_token = prompt_tokens
                output_token = completion_tokens

                self.total_tokens += total_tokens
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
                    'get_context': get_contexts,
                    'get_talk_nums': get_talk_nums,
                    'input_token': input_token,
                    'output_token': output_token,
                    'processing_time': processing_time,
                    'model': model_name
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

    def generate_response(self, question, context, character_prompt, get_talk_nums, task_name):
        try:
            start_time = time.time()

            # get_talk_numsをリストに変換
            talk_num_list = [num.strip() for num in get_talk_nums.split(',') if num.strip()]

            # 会話履歴を取得
            conversation_history = self.db.get_conversation_history(task_name, talk_num_list)

            # メッセージリストを構築
            messages = []

            # character_promptをsystemプロンプトとして追加
            messages.append({"role": "system", "content": character_prompt})

            # 会話履歴を時系列順に追加
            for convo in conversation_history:
                messages.append({"role": "user", "content": convo['user']})
                messages.append({"role": "assistant", "content": convo['assistant']})

            # 現在の質問を追加
            messages.append({"role": "user", "content": question})


            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=messages,
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0, 0, 0

            end_time = time.time()
            processing_time = end_time - start_time

            message = response['choices'][0]['message']['content'].strip()
            usage = response['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens

            return message, total_tokens, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {e}")
            return "", 0, 0, 0, 0

    def close(self):
        self.db.close()
