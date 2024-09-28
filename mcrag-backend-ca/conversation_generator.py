import openai
import json
import pandas as pd
import streamlit as st
import time
from database import ConversationDatabase  # データベースクラスをインポート

class ConversationGenerator:
    def __init__(self, temperature, api_key, task_name, words_info=None, db_file='conversation_data.db'):
        self.temperature = temperature
        self.api_key = api_key
        self.task_name = task_name
        self.words_info = words_info if words_info else {}
        self.total_tokens = 0
        self.total_processing_time = 0
        self.global_talk_num = 0  # 全体のtalk_numを管理
        openai.api_key = self.api_key  # APIキーの設定

        # データベースのセットアップ
        self.database = ConversationDatabase(db_file=db_file)

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

    def run_conversation(self, num_turns_per_word, aituber_prompt, user_prompt):
        all_conversations = []

        for word in self.words_info.keys():
            # Version 1: ワードを直接使用した会話の生成（内部で使用）
            conversation, tokens_v1, time_v1 = self.generate_conversation_with_word(
                word, num_turns_per_word, aituber_prompt, user_prompt
            )

            # Version 2: ワードを置換した会話の生成（ユーザーに提供）
            replaced_conversation, tokens_v2, time_v2 = self.replace_word_in_conversation(
                conversation, word
            )

            # 結果の累積（Version 2のみ）
            all_conversations.extend(replaced_conversation)
            self.total_tokens += tokens_v1 + tokens_v2
            self.total_processing_time += time_v1 + time_v2

        return all_conversations

    def generate_conversation_with_word(self, word, num_turns, aituber_prompt, user_prompt):
        conversation = []
        tokens = 0
        processing_time = 0
        words_remaining = set([word])
        words_used_directly = set()

        for turn in range(num_turns):
            self.global_talk_num += 1  # talk_numをインクリメント

            # ユーザーの発言生成
            user_messages = [
                {"role": "system", "content": user_prompt},
                {"role": "assistant", "content": f"これまでの会話履歴:\n{json.dumps(conversation, ensure_ascii=False, indent=2)}"},
            ]

            if words_remaining:
                # 初回のワード使用時
                user_messages.append({
                    "role": "user",
                    "content": f"""
あなたの発言には、以下のワードとその情報を自然に含めてください:
- ワード: {word}, 情報: {self.words_info[word]}
"""
                })
            else:
                # 二回目以降
                user_messages.append({
                    "role": "user",
                    "content": f"""
会話を続けてください。ただし、以下のワードを直接使用しないでください:
{word}
その代わりに、文脈に合った自然な言い換えや指示代名詞を使用してください。
"""
                })

            user_response, token_count_user, time_user = self.generate_message(user_messages)
            tokens += token_count_user
            processing_time += time_user

            conversation.append({
                "talk_num": self.global_talk_num,
                "task_name": self.task_name,
                "word": word,
                "user": user_response,
                "assistant": "",
                "token_count": token_count_user,
                "processing_time": time_user
            })

            # アシスタントの発言生成
            assistant_messages = [
                {"role": "system", "content": aituber_prompt},
                {"role": "assistant", "content": f"これまでの会話履歴:\n{json.dumps(conversation, ensure_ascii=False, indent=2)}"},
                {"role": "user", "content": user_response}
            ]

            if not words_remaining:
                # 二回目以降
                assistant_messages.append({
                    "role": "assistant",
                    "content": f"""
以下のワードを直接使用しないでください:
{word}
その代わりに、文脈に合った自然な言い換えや指示代名詞を使用してください。
"""
                })

            assistant_response, token_count_assistant, time_assistant = self.generate_message(assistant_messages)
            tokens += token_count_assistant
            processing_time += time_assistant

            conversation[-1]["assistant"] = assistant_response
            conversation[-1]["token_count"] += token_count_assistant
            conversation[-1]["processing_time"] += time_assistant

            # ワードの使用チェック
            if word in user_response or word in assistant_response:
                words_used_directly.add(word)
                words_remaining.discard(word)

        return conversation, tokens, processing_time

    def replace_word_in_conversation(self, conversation, word):
        replaced_conversation = []
        tokens = 0
        processing_time = 0
        word_used = False  # ワードが既に使用されたかを追跡

        for entry in conversation:
            replaced_entry = entry.copy()

            # ワードが既に使用されたかを確認
            if word_used:
                # 二回目以降の会話で置換を行う
                # ユーザー発言の置換
                user_context = replaced_entry["user"]
                user_replacement = self.get_replacement(word, user_context)
                replaced_entry["user"] = replaced_entry["user"].replace(word, user_replacement)

                # アシスタント発言の置換
                assistant_context = replaced_entry["assistant"]
                assistant_replacement = self.get_replacement(word, assistant_context)
                replaced_entry["assistant"] = replaced_entry["assistant"].replace(word, assistant_replacement)

                # トークン数と処理時間を更新
                tokens += 2 * 10  # get_replacementでの推定トークン数（簡易計算）
                processing_time += 2 * 0.5  # 推定処理時間（簡易計算）
            else:
                # 最初の会話では置換を行わないが、ワードが使用されたかを確認
                if word in replaced_entry["user"] or word in replaced_entry["assistant"]:
                    word_used = True  # 次回から置換を行う

            # temperatureとcreated_atを追加
            replaced_entry["temperature"] = self.temperature
            replaced_entry["created_at"] = self.database.get_current_timestamp()

            # データベースに保存
            self.database.insert_conversation(replaced_entry)

            replaced_conversation.append(replaced_entry)

        return replaced_conversation, tokens, processing_time

    def get_replacement(self, word, context):
        try:
            prompt = f"""
あなたは優れた日本語のライターです。以下の文脈と置換したいワードがあります。

文脈:
{context}

ワード:
{word}

このワードを直接使用せず、文脈に合った自然な言い換えや指示代名詞を一つ提案してください。提案のみを出力してください。
"""
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",  # 必要に応じてモデルを変更
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=self.temperature,
            )
            replacement = response['choices'][0]['message']['content'].strip()
            return replacement
        except Exception as e:
            st.error(f"置換語生成中にエラーが発生しました: {e}")
            return "それ"

    def get_conversation_dataframe(self, conversations):
        df = pd.DataFrame(conversations, columns=["talk_num", "task_name", "word", "user", "assistant", "token_count", "processing_time", "temperature", "created_at"])
        return df

    def close(self):
        self.database.close()
