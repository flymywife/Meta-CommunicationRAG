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

            # プライマリーキーの組み合わせを作成
            primary_key = (talk_nums_str, task_name, word)

            # 重複チェック
            if primary_key in existing_keys:
                st.error(f"プライマリーキーの重複エラーが発生しました。talk_nums: {talk_nums_str}, task_name: {task_name}, word: {word}")
                continue  # またはエラーを発生させて処理を停止する
            else:
                existing_keys.add(primary_key)

            conversation_text = self.format_conversation(chunk)

            # 質問の生成
            question_prompt = f"""
あなたは以下の会話履歴に基づいて、その内容に関する質問を1つ作成してください。
質問は、この会話履歴を全て参照しないと答えられないようにしてください。

会話履歴:
{conversation_text}

質問:
"""
            question, token_count_q, processing_time_q = self.generate_text(question_prompt)

            # 回答の生成
            answer_prompt = f"""
以下の会話履歴と質問があります。質問に対する回答を作成してください。

会話履歴:
{conversation_text}

質問:
{question}

回答:
"""
            answer, token_count_a, processing_time_a = self.generate_text(answer_prompt)

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

    def generate_text(self, prompt):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # 必要に応じてモデルを変更
                messages=[
                    {"role": "user", "content": prompt},
                ],
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

    def format_conversation(self, chunk):
        # 会話履歴をフォーマット
        conversation_text = ''
        for entry in chunk:
            conversation_text += f"ユーザー: {entry['user']}\n"
            conversation_text += f"アシスタント: {entry['assistant']}\n"
        return conversation_text

    def parse_csv_files(self, uploaded_files):
        # CSVファイルから会話履歴を読み込み、チャンクに分ける
        all_chunks = []
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                # 必要な列が存在するか確認
                required_columns = ['talk_num', 'task_name', 'word', 'user', 'assistant']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"{uploaded_file.name} に必要な列がありません。")
                    continue
                # talk_num ごとにソート
                df = df.sort_values('talk_num')
                # talk_num, task_name, word ごとにグループ化
                grouped = df.groupby(['task_name', 'word'])
                for _, group in grouped:
                    chunk = group.to_dict('records')
                    all_chunks.append(chunk)
            except Exception as e:
                st.error(f"{uploaded_file.name} の読み込み中にエラーが発生しました: {e}")
        return all_chunks
