# question_generator.py

import time
import json
from call_gpt import GPTClient  # GPTClientクラスをインポート
from database import ConversationDatabase, DataAlreadyExistsError

class DataNotFoundError(Exception):
    """データが見つからない場合の例外クラス"""
    pass

class QuestionGenerator:
    def __init__(self, temperature, api_key, db_file='conversation_data.db'):
        self.temperature = temperature
        self.api_key = api_key
        self.total_tokens = 0
        self.total_processing_time = 0

        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

        # データベースのインスタンスを作成
        self.db = ConversationDatabase(db_file=db_file)

    def generate_questions_and_answers(self, task_name):
        # 重複チェック: generated_qas テーブルに既に task_name が存在するか確認
        if self.db.has_generated_qas(task_name):
            raise DataAlreadyExistsError(f"タスク名 '{task_name}' の質問と回答は既に生成されています。")

        # データベースから会話履歴を取得
        conversation_entries = self.get_conversation_entries(task_name)

        if not conversation_entries:
            raise DataNotFoundError(f"タスク名 '{task_name}' に対応するデータが見つかりません。")

        results = []
        existing_keys = set()  # 重複チェック用のセット

        for entry in conversation_entries:
            talk_num = entry['talk_num']
            word = entry.get('word', '')
            character_prompt = entry.get('character_prompt', '')
            user_prompt = entry.get('user_prompt', '')
            word_info_id = entry['word_info_id']
            task_id = entry['task_id']

            # プライマリーキーの組み合わせを作成
            primary_key = (talk_num, task_name, word)

            # 重複チェック
            if primary_key in existing_keys:
                print(f"プライマリーキーの重複エラーが発生しました。talk_num: {talk_num}, task_name: {task_name}, word: {word}")
                continue
            else:
                existing_keys.add(primary_key)

            # before_user と before_assistant が空文字でない場合のみ対象とする
            if entry['before_user'] and entry['before_assistant']:
                # 会話履歴をメッセージ形式で取得
                conversation_messages = self.get_conversation_messages(entry)

                # 質問の生成
                question, token_count_q, processing_time_q = self.generate_question(
                    conversation_messages, user_prompt
                )

                # 回答の生成
                answer, token_count_a, processing_time_a = self.generate_answer(
                    conversation_messages, question, character_prompt
                )

                # 結果の保存
                total_token_count = token_count_q + token_count_a
                total_processing_time = processing_time_q + processing_time_a
                self.total_tokens += total_token_count
                self.total_processing_time += total_processing_time

                result_entry = {
                    'talk_nums': talk_num,
                    'task_name': task_name,
                    'word': word,
                    'question': question.strip(),
                    'answer': answer.strip(),
                    'token_count': total_token_count,
                    'processing_time': total_processing_time,
                    'task_id': task_id,
                    'word_info_id': word_info_id
                }
                results.append(result_entry)

                # データベースに保存
                try:
                    self.save_generated_qa(result_entry)
                except DataAlreadyExistsError as e:
                    raise e

        return results

    def save_generated_qa(self, result_entry):
        db_entry = {
            'task_name': result_entry['task_name'],
            'task_id': result_entry['task_id'],
            'word_info_id': result_entry['word_info_id'],
            'talk_nums': result_entry['talk_nums'],
            'question': result_entry['question'],
            'answer': result_entry['answer'],
            'token_count': result_entry['token_count'],
            'processing_time': result_entry['processing_time']
        }

        self.db.insert_generated_qa(db_entry)

    def generate_question(self, conversation_messages, user_prompt):
        try:
            start_time = time.time()
            # システムプロンプト
            system_prompt = f"""
以下のユーザー設定に基づいて、与えられた会話内容に関する質問を作成してください。

ユーザー設定:
{user_prompt}

以下の指示に従って、会話内容に関する質問を一つ生成してください。

- 質問は必ず上記のユーザー設定の口調で作成してください。
- 質問は、与えられた会話内容を参照しなければ答えられない質問にしてください。
- 独創的な質問はNGです。必ず与えられた会話内容から答えられる質問で、明確な回答が得られる質問にしてください。
- 質問は具体的で明確であり、できるだけ簡潔な質問にしてください。
- キャラクターの設定による口調を反映させ、自然な会話の流れを維持してください。
"""

            # Function Callingの定義
            functions = [
                {
                    "name": "generate_question",
                    "description": "会話内容に関する質問を生成します。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "生成された質問。"
                            }
                        },
                        "required": ["question"]
                    }
                }
            ]

            # GPT呼び出し
            response = self.gpt_client.call_gpt_function(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_messages,
                    {"role": "user", "content": "上記の会話内容に関する質問を一つ作成してください。"}
                ],
                functions=functions,
                function_call={"name": "generate_question"},
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0

            function_response = response["choices"][0]["message"].get("function_call")
            if not function_response:
                print("Function Callingのレスポンスが得られませんでした。")
                return "", 0, 0

            arguments = json.loads(function_response.get("arguments", "{}"))
            question = arguments.get("question", "")
            token_count = response['usage']['total_tokens']
            end_time = time.time()
            processing_time = end_time - start_time

            return question, token_count, processing_time

        except Exception as e:
            print(f"質問生成中にエラーが発生しました: {e}")
            return "", 0, 0

    def generate_answer(self, conversation_messages, question, character_prompt):
        try:
            start_time = time.time()
            # システムプロンプト
            system_prompt = f"""
あなたは以下のキャラクター設定に基づいて、質問に回答する専門家です。

キャラクター設定:
{character_prompt}

以下の指示に従って、質問に対する回答を生成してください。

- 回答は必ず上記のキャラクター設定の口調で作成してください。
- 回答は、提供された会話内容に基づいて、質問の全ての部分にしっかりと答えてください。
- 会話内容を参照し、正確かつ詳細な情報を提供してください。
- キャラクターの性格や背景を反映させ、自然な会話の流れを維持してください。
"""

            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_messages,
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0

            message = response["choices"][0]["message"]
            answer = message.get('content', '').strip()
            token_count = response['usage']['total_tokens']
            end_time = time.time()
            processing_time = end_time - start_time

            return answer, token_count, processing_time

        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {e}")
            print(f"エラー詳細: {e}")
            return "", 0, 0

    def get_conversation_messages(self, entry):
        # 会話履歴をメッセージ形式に変換
        messages = []

        # 置換前の発言を含める
        messages.append({"role": "user", "content": entry['before_user']})
        messages.append({"role": "assistant", "content": entry['before_assistant']})

        return messages

    def get_conversation_entries(self, task_name):
        # データベースから指定された task_name の会話履歴を取得
        try:
            # データベースのメソッドを呼び出してデータを取得
            conversations = self.db.get_conversations_with_task_name(task_name)

            if not conversations:
                print(f"タスク名 '{task_name}' に対応する会話が見つかりませんでした。")
                return []

            # before_user と before_assistant が空文字でないエントリのみ取得
            valid_entries = [entry for entry in conversations if entry['before_user'] and entry['before_assistant']]

            return valid_entries
        except Exception as e:
            print(f"会話エントリの取得中にエラーが発生しました: {e}")
            return []

    def generate_json_for_download(self, task_name):
        # 新しいテーブルからデータを取得して、JSON を生成
        qas_list = self.db.get_generated_qas_by_task_name(task_name)
        return qas_list

    def close(self):
        self.db.close()
