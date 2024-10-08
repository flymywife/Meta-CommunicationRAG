# answer_evaluator.py

import streamlit as st
import time
import json
from call_gpt import GPTClient  # GPTClientクラスをインポート

class AnswerEvaluator:
    def __init__(self, temperature, api_key):
        self.temperature = temperature
        self.api_key = api_key
        self.total_tokens = 0
        self.total_processing_time = 0

        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

    def evaluate_answers(self, json_data):
        results = []
        # JSONデータをリストとして扱う
        data_list = json_data if isinstance(json_data, list) else [json_data]

        for idx, entry in enumerate(data_list):
            talk_nums = entry.get('talk_nums', '')
            task_name = entry.get('task_name', '')
            word = entry.get('word', '')
            query = entry.get('query', '')
            expected_answer = entry.get('answer', '')

            # GPTにクエリを送信
            response_text, token_count, processing_time = self.generate_response(query)

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
            # GPT呼び出し
            response = self.gpt_client.call_gpt(
                messages=[
                    {"role": "user", "content": query},
                ],
                max_tokens=500,
                temperature=self.temperature,
            )

            if not response:
                return "", 0, 0

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
