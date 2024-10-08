# conversation_generator.py

import json
import pandas as pd
import streamlit as st
import time
import re
from database import ConversationDatabase  # データベースクラスをインポート
from call_gpt import GPTClient  # GPTClientクラスをインポート

class ConversationGenerator:
    def __init__(self, temperature, api_key, task_name, words_info, character_prompt, user_prompt, db_file='conversation_data.db'):
        self.temperature = temperature
        self.api_key = api_key
        self.task_name = task_name
        self.words_info = words_info if words_info else []
        self.character_prompt = character_prompt
        self.user_prompt = user_prompt
        self.total_tokens = 0
        self.total_processing_time = 0
        self.global_talk_num = 0  # 全体のtalk_numを管理

        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

        # データベースのセットアップ
        self.database = ConversationDatabase(db_file=db_file)
        # タスクをデータベースに挿入
        self.task_id = self.database.insert_task({
            'task_name': self.task_name,
            'character_prompt': self.character_prompt,
            'user_prompt': self.user_prompt
        })

        if self.task_id is None:
            raise Exception("タスクIDが取得できませんでした。データベースへのタスクの挿入に失敗した可能性があります。")

        # words_infoをデータベースに挿入
        self.insert_words_info()

    def insert_words_info(self):
        self.word_info_list = []  # ワード情報のリスト（word, infos, info, word_info_id）
        for word_info in self.words_info:
            word = word_info['word']
            infos = word_info['infos']
            # wordに括弧や日本語の引用符が含まれているかチェック
            if self.contains_invalid_characters(word):
                # エラーを発生させる
                raise ValueError(f"ワード '{word}' に無効な文字（括弧や日本語の引用符）が含まれています。処理を中止します。")
            # infosを結合して一つのinfoにする
            info = ' '.join(infos)
            word_info_id = self.database.insert_word_info(self.task_id, word, info)
            if word_info_id is not None:
                # infosを含めてword_info_listに追加
                self.word_info_list.append({
                    'word': word,
                    'infos': infos,
                    'info': info,
                    'word_info_id': word_info_id
                })

    def contains_invalid_characters(self, word):
        """
        ワードに括弧 () や日本語の引用符 「」 が含まれているかをチェックします。
        含まれている場合は True を返します。
        """
        invalid_characters = ['(', ')', '（', '）', '「', '」']
        for char in invalid_characters:
            if char in word:
                return True
        return False

    def generate_message(self, messages):
        try:
            start_time = time.time()
            response = self.gpt_client.call_gpt(
                messages=messages,
                max_tokens=500,
                temperature=self.temperature
            )
            end_time = time.time()
            processing_time = end_time - start_time

            if not response:
                return "", 0, 0

            message = response['choices'][0]['message']['content'].strip()
            token_count = response['usage']['total_tokens']

            return message, token_count, processing_time
        except Exception as e:
            st.error(f"メッセージ生成中にエラーが発生しました: {e}")
            return "", 0, 0

    def run_conversation(self):
        all_conversations = []

        for word_info in self.word_info_list:
            word = word_info['word']
            infos = word_info['infos']
            word_info_id = word_info['word_info_id']

            # 会話履歴を初期化
            conversation_history = []

            # 各情報に対して会話を生成
            conversation, tokens, processing_time = self.generate_conversation_with_word(
                word, infos, word_info_id, conversation_history
            )

            # 結果の累積
            all_conversations.extend(conversation)
            self.total_tokens += tokens
            self.total_processing_time += processing_time

        return all_conversations

    def generate_conversation_with_word(self, word, infos, word_info_id, conversation_history):
        conversation = []
        tokens = 0
        processing_time = 0
        word_used = False  # ワードが既に使用されたかを追跡

        for info_index, info in enumerate(infos):
            if not info.strip():
                continue  # 情報が空の場合はスキップ

            # これまでの会話履歴を取得
            conversation_messages = self.convert_conversation_to_messages(conversation_history)

            # ユーザー発言にワードと情報を含める
            user_content = f"""
Use the following words as the subject of the conversation and use the word information provided to start a natural conversation.

ワード: {word}
情報: {info}
"""

            user_messages = [
                {"role": "system", "content": self.user_prompt},
                *conversation_messages,
                {"role": "user", "content": user_content}
            ]

            user_response, token_count_user, time_user = self.generate_message(user_messages)
            tokens += token_count_user
            processing_time += time_user

            # ワードがユーザー発言に含まれているか確認
            if not word_used and self.word_in_text(word, user_response):
                word_used = True

            # アシスタントの発言生成
            assistant_messages = [
                {"role": "system", "content": self.character_prompt},
                *conversation_messages,
                {"role": "user", "content": user_response}
            ]

            assistant_response, token_count_assistant, time_assistant = self.generate_message(assistant_messages)
            tokens += token_count_assistant
            processing_time += time_assistant

            # 二回目以降の会話でワードを置換
            if word_used and info_index >= 1:
                # Function Calling を使用して置換語を取得し、ユーザーとアシスタントの発言を置換
                user_response = self.replace_word_in_text(word, user_response)
                assistant_response = self.replace_word_in_text(word, assistant_response)

            self.global_talk_num += 1  # talk_numをインクリメント

            # ユーザーの発言を会話に追加
            conversation_entry = {
                "talk_num": str(self.global_talk_num),
                "task_name": self.task_name,
                "word": word,
                "info": info,
                "user": user_response,
                "assistant": assistant_response,
                "token_count": str(token_count_user + token_count_assistant),
                "processing_time": str(time_user + time_assistant),
                "temperature": str(self.temperature),
                "created_at": self.database.get_current_timestamp(),
                "info_num": info_index + 1
            }

            conversation.append(conversation_entry)
            conversation_history.append(conversation_entry)

            # データベースに保存
            entry = conversation_entry.copy()
            entry['task_id'] = str(self.task_id)
            entry['word_info_id'] = word_info_id
            self.database.insert_conversation(entry)

        return conversation, tokens, processing_time

    def replace_word_in_text(self, word, text):
        # Function Calling を使用して置換語を取得
        replacement_info = self.get_replacement_via_function_calling(word, text)
        replacement = replacement_info.get('replacement', 'それ')

        # ワードのバリエーションを生成
        word_variants = [word.strip()]
        # 括弧や引用符が含まれないため、この部分は不要になりました

        # 各バリエーションで置換
        for variant in word_variants:
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def get_replacement_via_function_calling(self, word, context):
        try:
            # ワードのバリエーションを生成
            word_variants = [word.strip()]
            # 括弧や引用符が含まれないため、この部分は不要になりました

            # バリエーションをコンマ区切りで結合
            word_list = ', '.join(f"'{w}'" for w in word_variants)

            # システムプロンプト
            system_prompt = "You are an assistant that provides a replacement for given words in the context, using demonstrative pronouns or natural paraphrases in Japanese."

            user_content = f"""
Replace the word {word_list} in the following context with a demonstrative pronoun or a natural paraphrase in Japanese. Provide only the replacement word.

Context:
{context}
"""

            # Function Callingの定義
            functions = [
                {
                    "name": "replace_word",
                    "description": "Provides a replacement for the specified words.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "replacement": {
                                "type": "string",
                                "description": "The replacement word or pronoun in Japanese."
                            }
                        },
                        "required": ["replacement"]
                    }
                }
            ]

            # GPT呼び出し
            response = self.gpt_client.call_gpt_function(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                functions=functions,
                function_call={"name": "replace_word"},
                max_tokens=50,
                temperature=self.temperature,
            )

            if not response:
                return {"replacement": "それ"}

            function_response = response["choices"][0]["message"].get("function_call")
            if not function_response:
                st.error("Function Callingのレスポンスが得られませんでした。")
                return {"replacement": "それ"}

            arguments = json.loads(function_response.get("arguments", "{}"))
            replacement = arguments.get("replacement", "それ")
            return {
                "replacement": replacement
            }

        except Exception as e:
            st.error(f"置換語生成中にエラーが発生しました: {e}")
            return {
                "replacement": "それ"
            }

    def word_in_text(self, word, text):
        # ワードのバリエーションを生成
        word_variants = [word.strip()]
        # 括弧や引用符が含まれないため、この部分は不要になりました

        # 各バリエーションで検索
        for variant in word_variants:
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            if pattern.search(text):
                return True
        return False

    def convert_conversation_to_messages(self, conversation):
        messages = []
        for entry in conversation:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        return messages

    def close(self):
        self.database.close()
