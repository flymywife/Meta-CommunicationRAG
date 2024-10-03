# question_generator.py

import openai
import pandas as pd
import streamlit as st
import time
import json

class QuestionGenerator:
    def __init__(self, temperature, api_key):
        self.temperature = temperature
        self.api_key = api_key
        self.total_tokens = 0
        self.total_processing_time = 0
        openai.api_key = self.api_key  # APIキーの設定

    def generate_question_and_answer(self, conversation_chunks):
        results = []
        existing_keys = set()  # 重複チェック用のセット

        for chunk in conversation_chunks:
            talk_nums = [str(entry['talk_num']) for entry in chunk]
            talk_nums_str = ','.join(talk_nums)
            task_name = chunk[0]['task_name'] if 'task_name' in chunk[0] else ''
            word = chunk[0]['word'] if 'word' in chunk[0] else ''
            character_prompt = chunk[0]['character_prompt'] if 'character_prompt' in chunk[0] else ''
            user_prompt = chunk[0]['user_prompt'] if 'user_prompt' in chunk[0] else ''

            # プライマリーキーの組み合わせを作成
            primary_key = (talk_nums_str, task_name, word)

            # 重複チェック
            if primary_key in existing_keys:
                st.error(f"プライマリーキーの重複エラーが発生しました。talk_nums: {talk_nums_str}, task_name: {task_name}, word: {word}")
                continue  # またはエラーを発生させて処理を停止する
            else:
                existing_keys.add(primary_key)

            # 会話履歴をメッセージ形式で取得
            conversation_messages = self.get_conversation_messages(chunk)

            # 質問の生成
            # ユーザーのプロンプトをシステムメッセージとして設定
            # 会話履歴を含める
            # 最後に質問生成の指示を含める

            question_messages = [
                {"role": "system", "content": user_prompt}
            ] + conversation_messages + [
                {"role": "user", "content": "上記の会話内容に基づいて、その内容に関する質問を1つ作成してください。質問は、この会話を全て参照しないと答えられないようにしてください。"}
            ]

            question, token_count_q, processing_time_q = self.generate_text(question_messages)

            # 回答の生成
            # キャラクタープロンプトをシステムメッセージとして設定
            # 会話履歴を含める
            # 質問をユーザーからのメッセージとして含める

            answer_messages = [
                {"role": "system", "content": character_prompt}
            ] + conversation_messages + [
                {"role": "user", "content": question}
            ]

            answer, token_count_a, processing_time_a = self.generate_text(answer_messages)

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

    def generate_text(self, messages):
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
            st.error(f"テキスト生成中にエラーが発生しました: {e}")
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
            from operator import itemgetter

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
