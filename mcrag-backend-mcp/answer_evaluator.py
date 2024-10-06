import openai
import pandas as pd
import streamlit as st
import time
import json

class AnswerEvaluator:
    def __init__(self, temperature, api_key):
        self.temperature = temperature
        self.api_key = api_key
        self.total_tokens = 0
        self.total_processing_time = 0
        openai.api_key = self.api_key  # APIキーの設定

    def evaluate_answers(self, df):
        results = []
        for idx, row in df.iterrows():
            talk_nums = row['talk_nums']
            task_name = row['task_name']
            word = row['word']
            query = row['query']
            expected_answer = row['answer']

            # GPTにクエリを送信
            start_time = time.time()
            response_text, token_count, processing_time = self.generate_response(query)
            end_time = time.time()

            # 回答の比較
            is_correct = self.compare_answers(expected_answer, response_text)

            # 結果の保存
            self.total_tokens += token_count
            self.total_processing_time += processing_time

            results.append({
                'talk_nums': talk_nums,
                'task_name': task_name,
                'word': word,
                'query': query,
                'expected_answer': expected_answer,
                'gpt_response': response_text,
                'is_correct': int(is_correct),  # 1 or 0
                'token_count': token_count,
                'processing_time': processing_time
            })
        return results

    def generate_response(self, query):
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",  # 必要に応じてモデルを変更
                messages=[
                    {"role": "user", "content": query},
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
            st.error(f"回答生成中にエラーが発生しました: {e}")
            return "", 0, 0

    def compare_answers(self, expected_answer, actual_answer):
        # 簡易的な比較（完全一致）
        # 必要に応じて類似度計算や高度な比較を行う
        return expected_answer.strip() == actual_answer.strip()
