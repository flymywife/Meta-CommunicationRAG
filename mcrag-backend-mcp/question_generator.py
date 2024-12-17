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
                question, q_total_tokens, q_input_tokens, q_output_tokens, processing_time_q = self.generate_question(
                    conversation_messages, user_prompt, word
                )

                # 回答の生成
                answer, a_total_tokens, a_input_tokens, a_output_tokens, processing_time_a = self.generate_answer(
                    conversation_messages, question, character_prompt
                )

                # トークン数を分離
                input_token = q_input_tokens + a_input_tokens
                output_token = q_output_tokens + a_output_tokens

                total_token_count = q_total_tokens + a_total_tokens
                total_processing_time = processing_time_q + processing_time_a
                self.total_tokens += total_token_count
                self.total_processing_time += total_processing_time

                result_entry = {
                    'talk_nums': talk_num,
                    'task_name': task_name,
                    'word': word,
                    'question': question.strip(),
                    'answer': answer.strip(),
                    'input_token': input_token,      # 新規追加
                    'output_token': output_token,    # 新規追加
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
            'input_token': result_entry['input_token'],   # 新規追加
            'output_token': result_entry['output_token'], # 新規追加 
            'processing_time': result_entry['processing_time']
        }

        self.db.insert_generated_qa(db_entry)

    def generate_question(self, conversation_messages, user_prompt, word):
        try:
            start_time = time.time()
            # システムプロンプト
            system_prompt = f"""
以下のキャラクター設定に基づき、会話履歴の情報だけで回答可能な簡単な質問を作成します。

キャラクター設定:
{user_prompt}
- 質問の主語には必ず{word}を使用してください。
"""

            query = f"""
以下の手順に従って、会話の内容に関する質問を1つ作成してください。

- 質問は、会話の履歴を参照すれば誰でも簡単に答えられるもの、考えなくても答えられるもの。
- 会話の流れは自然で、登場人物の設定を反映させること。
- 会話履歴から連想・考察が必要な質問はNGです。例えば
・「どう思う？」はダメ
・「その場合どうする？」はダメ
・「あなたはどう考える？」はダメ
などの考察が必要な質問はダメ。
良い質問は正誤がはっきりと会話履歴を読めば誰でもわかる質問。
- 会話履歴の中で名詞がある場合はそれを質問に積極的に取り入れること。
"""
            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_messages,
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0, 0, 0

            message = response["choices"][0]["message"]
            question = message.get('content', '').strip()
            usage = response['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            end_time = time.time()
            processing_time = end_time - start_time

            return question, total_tokens, prompt_tokens, completion_tokens, processing_time

        except Exception as e:
            print(f"質問生成中にエラーが発生しました: {e}")
            return "", 0, 0, 0, 0

    def generate_answer(self, conversation_messages, question, character_prompt):
        try:
            start_time = time.time()
            # システムプロンプト
            system_prompt = f"""{character_prompt}"""

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
                return "", 0, 0, 0, 0

            message = response["choices"][0]["message"]
            answer = message.get('content', '').strip()
            usage = response['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            end_time = time.time()
            processing_time = end_time - start_time

            return answer, total_tokens, prompt_tokens, completion_tokens, processing_time

        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {e}")
            print(f"エラー詳細: {e}")
            return "", 0, 0, 0, 0

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
