# evaluation_page.py

import streamlit as st
import pandas as pd
import requests
import json

def show_evaluation_page(api_key, temperature):
    st.title("回答の評価")

    # タスク名の入力
    task_name = st.text_input("タスク名を入力してください:")

    # 評価ボタン
    if st.button("回答を評価"):
        if not api_key or api_key == "your_api_key":
            st.error("有効なOpenAI APIキーを入力してください。")
        elif not task_name:
            st.error("タスク名を入力してください。")
        else:
            try:
                with st.spinner("回答を評価しています..."):
                    # リクエストボディの作成
                    payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "task_name": task_name
                    }

                    # バックエンドAPIのエンドポイント（ローカルホスト）
                    url = "http://localhost:8000/evaluate_answers"

                    # APIリクエストの送信
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        results = data["results"]
                        total_tokens = data.get("total_tokens", 0)
                        total_processing_time = data.get("total_processing_time", 0)

                        # 結果の表示
                        st.subheader("評価結果")
                        for result in results:
                            st.write(f"**Talk Nums**: {result.get('talk_nums', '')}")
                            st.write(f"**Task Name**: {result.get('task_name', '')}")
                            st.write(f"**Word**: {result.get('word', '')}")
                            st.write(f"**Query**: {result.get('question', '')}")
                            st.write(f"**Expected Answer**: {result.get('expected_answer', '')}")
                            st.write(f"**GPT Response**: {result.get('gpt_response', '')}")
                            st.write(f"**Get Context**: {result.get('get_context', '')}")
                            st.write(f"**Get Talk Nums**: {result.get('get_talk_nums', '')}")
                            st.write(f"**Token Count**: {result.get('token_count', 0)}")
                            st.write(f"**Processing Time**: {result.get('processing_time', 0):.2f} seconds")
                            st.write(f"**Model**: {result.get('model', '')}")
                            st.write("---")

                        # JSONデータの作成
                        json_data = json.dumps(results, ensure_ascii=False, indent=2).encode('utf-8')
                        st.download_button(
                            label="評価結果をJSONでダウンロード",
                            data=json_data,
                            file_name=f"evaluation_results_{task_name}.json",
                            mime='application/json',
                        )
                    else:
                        # エラーメッセージを表示
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except json.JSONDecodeError:
                            error_detail = response.text
                        st.error(f"エラーが発生しました: {error_detail}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
