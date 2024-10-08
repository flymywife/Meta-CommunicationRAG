# question_generator.py

import pandas as pd
import streamlit as st
import time
import json
from call_gpt import GPTClient  # GPTClientクラスをインポート

class QuestionGenerator:
    def __init__(self, temperature, api_key):
        self.temperature = temperature
        self.api_key = api_key
        self.total_tokens = 0
        self.total_processing_time = 0

        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

    def generate_question_and_answer(self, conversation_chunks):
        results = []
        existing_keys = set()  # 重複チェック用のセット

        for chunk in conversation_chunks:
            talk_nums = [str(entry['talk_num']) for entry in chunk]
            talk_nums_str = ','.join(talk_nums)
            task_name = chunk[0].get('task_name', '')
            word = chunk[0].get('word', '')
            character_prompt = chunk[0].get('character_prompt', '')
            user_prompt = chunk[0].get('user_prompt', '')

            # プライマリーキーの組み合わせを作成
            primary_key = (talk_nums_str, task_name, word)

            # 重複チェック
            if primary_key in existing_keys:
                st.error(f"プライマリーキーの重複エラーが発生しました。talk_nums: {talk_nums_str}, task_name: {task_name}, word: {word}")
                continue
            else:
                existing_keys.add(primary_key)

            # 会話履歴をメッセージ形式で取得
            conversation_messages = self.get_conversation_messages(chunk)

            # 質問の生成（Function Calling を使用）
            question, token_count_q, processing_time_q = self.generate_question_via_function_calling(
                conversation_messages, user_prompt
            )

            # 回答の生成（Function Calling を使用せず、プロンプトエンジニアリングを使用）
            answer, token_count_a, processing_time_a = self.generate_answer(
                conversation_messages, question, character_prompt
            )

            # 結果の保存
            total_token_count = token_count_q + token_count_a
            total_processing_time = processing_time_q + processing_time_a
            self.total_tokens += total_token_count
            self.total_processing_time += total_processing_time

            results.append({
                'talk_nums': talk_nums_str,
                'task_name': task_name,
                'word': word,
                'query': question.strip(),
                'answer': answer.strip(),
                'token_count': total_token_count,
                'processing_time': total_processing_time
            })
        return results

    def generate_question_via_function_calling(self, conversation_messages, user_prompt):
        try:
            start_time = time.time()
            # システムプロンプト
            system_prompt = f"""
あなたは以下のキャラクター設定に基づいて、与えられた会話内容に関する質問を作成する専門家です。

キャラクター設定:
{user_prompt}

以下の指示に従って、会話内容に関する質問を一つ生成してください。

- 質問は必ず上記のキャラクター設定の口調で作成してください。
- 質問は、会話の全体的な内容や複数の話題を関連付けたものにしてください。
- 質問に答えるためには、会話の全体を参照する必要があるようにしてください。
- 単純な yes/no で答えられる質問は避け、より深い思考や分析を要する質問を心がけてください。
- 可能であれば、会話で触れられた複数のトピックを結びつけるような質問を作成してください。
- 質問は具体的で明確であり、曖昧さを避けてください。
- 質問の難易度は中級から上級レベルにし、会話の内容を十分に理解していないと答えられないようにしてください。
- キャラクターの性格や背景を反映させ、自然な会話の流れを維持してください。
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
                max_tokens=500,
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0

            function_response = response["choices"][0]["message"].get("function_call")
            if not function_response:
                st.error("Function Callingのレスポンスが得られませんでした。")
                return "", 0, 0

            arguments = json.loads(function_response.get("arguments", "{}"))
            question = arguments.get("question", "")
            token_count = response['usage']['total_tokens']
            end_time = time.time()
            processing_time = end_time - start_time

            return question, token_count, processing_time

        except Exception as e:
            st.error(f"質問生成中にエラーが発生しました: {e}")
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
    - 回答は、会話履歴の内容に基づいて、質問の全ての部分にしっかりと答えてください。
    - 会話履歴を参照し、正確かつ詳細な情報を提供してください。
    - キャラクターの性格や背景を反映させ、自然な会話の流れを維持してください。
    """

            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_messages,
                    {"role": "user", "content": question}
                ],
                max_tokens=1000,
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
            st.error(f"回答生成中にエラーが発生しました: {e}")
            print(f"エラー詳細: {e}")
            return "", 0, 0

    def get_conversation_messages(self, chunk):
        # 会話履歴をメッセージ形式に変換
        messages = []
        for entry in chunk:
            messages.append({"role": "user", "content": entry['user']})
            messages.append({"role": "assistant", "content": entry['assistant']})
        return messages

    def parse_json_data(self, json_data):
        # JSONデータから会話履歴を読み込み、チャンクに分ける
        all_chunks = []
        try:
            conversations = json_data.get('conversations', [])
            character_prompt = json_data.get('character_prompt', '')
            user_prompt = json_data.get('user_prompt', '')
            task_name = json_data.get('task_name', '')

            # talk_num, task_name, word ごとにグループ化
            from itertools import groupby

            # conversations を task_name, word, talk_num でソート
            conversations = sorted(conversations, key=lambda x: (x['task_name'], x['word'], int(x['talk_num'])))

            # groupby でグループ化
            for (task_name_key, word_key), group in groupby(conversations, key=lambda x: (x['task_name'], x['word'])):
                chunk = list(group)
                # 各エントリにキャラクタープロンプトとユーザープロンプトを追加
                for entry in chunk:
                    entry['character_prompt'] = character_prompt
                    entry['user_prompt'] = user_prompt
                all_chunks.append(chunk)
        except Exception as e:
            st.error(f"JSONデータの解析中にエラーが発生しました: {e}")
        return all_chunks
