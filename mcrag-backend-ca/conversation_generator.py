# conversation_generator.py

import openai
import json
import pandas as pd
import streamlit as st
import time
import re
from database import ConversationDatabase  # データベースクラスをインポート

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
        openai.api_key = self.api_key  # APIキーの設定

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
        self.word_info_list = []  # ワード情報のリスト（word, info, word_info_id）
        for word_info in self.words_info:
            word = word_info['word']
            info = word_info['info']
            word_info_id = self.database.insert_word_info(self.task_id, word, info)
            if word_info_id is not None:
                self.word_info_list.append({'word': word, 'info': info, 'word_info_id': word_info_id})

    def generate_message(self, messages):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",  # 必要に応じてモデルを変更
                messages=messages,
                max_tokens=500,
                temperature=self.temperature,
            )
            end_time = time.time()
            processing_time = end_time - start_time

            message = response['choices'][0]['message']['content'].strip()
            token_count = response['usage']['total_tokens']

            return message, token_count, processing_time
        except Exception as e:
            st.error(f"メッセージ生成中にエラーが発生しました: {e}")
            return "", 0, 0

    def run_conversation(self, num_turns_per_word):
        all_conversations = []

        for word_info in self.word_info_list:
            word = word_info['word']
            info = word_info['info']
            word_info_id = word_info['word_info_id']

            # 会話の生成
            conversation, tokens, processing_time = self.generate_conversation_with_word(
                word, info, num_turns_per_word, word_info_id
            )

            # 結果の累積
            all_conversations.extend(conversation)
            self.total_tokens += tokens
            self.total_processing_time += processing_time

        return all_conversations

    def generate_conversation_with_word(self, word, info, num_turns, word_info_id):
        conversation = []
        tokens = 0
        processing_time = 0
        word_used = False  # ワードが既に使用されたかを追跡

        for turn in range(num_turns):
            self.global_talk_num += 1  # talk_numをインクリメント

            # これまでの会話履歴を取得
            conversation_messages = self.convert_conversation_to_messages(conversation)

            # ユーザーの発言生成
            if not word_used:
                # 初回のユーザー発言にワードと情報を含める
                user_content = f"""
以下のワードとその情報を使って、自然な会話を始めてください。

ワード: {word}
情報: {info}
"""
            else:
                # 二回目以降は会話を続ける
                user_content = "会話を続けてください。"

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
            if word_used and turn >= 1:
                # Function Calling を使用して置換語を取得し、ユーザーとアシスタントの発言を置換
                user_response = self.replace_word_in_text(word, user_response)
                assistant_response = self.replace_word_in_text(word, assistant_response)

            # ユーザーの発言を会話に追加
            conversation.append({
                "talk_num": str(self.global_talk_num),
                "task_name": self.task_name,
                "word": word,
                "info": info,
                "user": user_response,
                "assistant": assistant_response,
                "token_count": str(token_count_user + token_count_assistant),
                "processing_time": str(time_user + time_assistant),
                "temperature": str(self.temperature),
                "created_at": self.database.get_current_timestamp()
            })

            # データベースに保存
            entry = conversation[-1].copy()
            entry['task_id'] = str(self.task_id)
            entry['word_info_id'] = word_info_id
            self.database.insert_conversation(entry)

        return conversation, tokens, processing_time

    def replace_word_in_text(self, word, text):
        # Function Calling を使用して置換語を取得
        replacement_info = self.get_replacement_via_function_calling(word, text)
        replacement = replacement_info.get('replacement', 'それ')

        # ワードのバリエーションを生成
        word_variants = [word]
        if '(' in word and ')' in word:
            # 括弧内と括弧外のワードを取得
            word_without_parens = re.sub(r'\s*\(.*?\)\s*', '', word).strip()
            word_inside_parens = re.search(r'\((.*?)\)', word).group(1).strip()
            word_variants.extend([word_without_parens, word_inside_parens])
        else:
            word_variants.append(word.strip())

        # 各バリエーションで置換
        for variant in word_variants:
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text


    def get_replacement_via_function_calling(self, word, context):
        try:
            # ワードのバリエーションを生成
            word_variants = [word]
            if '(' in word and ')' in word:
                # 括弧内と括弧外のワードを取得
                word_without_parens = re.sub(r'\s*\(.*?\)\s*', '', word).strip()
                word_inside_parens = re.search(r'\((.*?)\)', word).group(1).strip()
                word_variants.extend([word_without_parens, word_inside_parens])
            else:
                word_variants.append(word.strip())

            # バリエーションをコンマ区切りで結合
            word_list = ', '.join(f"'{w}'" for w in word_variants)

            # Function Calling を使用して置換語を取得
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",  # Function Calling に対応したモデル
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that provides a replacement for given words in the context, using demonstrative pronouns or natural paraphrases in Japanese."
                    },
                    {
                        "role": "user",
                        "content": f"""
    Replace the words {word_list} in the following context with a demonstrative pronoun or a natural paraphrase in Japanese. Provide only the replacement word.

    Context:
    {context}
    """
                    }
                ],
                functions=[
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
                ],
                function_call={"name": "replace_word"},
                max_tokens=50,
                temperature=self.temperature,
            )

            function_response = response["choices"][0]["message"]["function_call"]
            arguments = json.loads(function_response["arguments"])
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
        word_variants = [word]
        if '(' in word and ')' in word:
            # 括弧内と括弧外のワードを取得
            word_without_parens = re.sub(r'\s*\(.*?\)\s*', '', word).strip()
            word_inside_parens = re.search(r'\((.*?)\)', word).group(1).strip()
            word_variants.extend([word_without_parens, word_inside_parens])
        else:
            word_variants.append(word.strip())

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
