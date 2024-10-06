import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import json

def show_evaluation_page():
    # 環境変数のロード
    load_dotenv()
    default_api_key = os.getenv("OPENAI_API_KEY", "your_api_key")

    st.title("回答の評価")

    # サイドバーの設定
    st.sidebar.title("設定")

    # OpenAI APIキーの入力欄
    api_key = st.sidebar.text_input("OpenAI APIキーを入力してください:", value=default_api_key)

    # Temperatureのスライダー
    temperature = st.sidebar.slider("Temperatureを選択してください:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # CSVファイルのアップロード
    uploaded_file = st.file_uploader("質問と回答のCSVファイルをアップロードしてください:", type="csv")

    # 評価ボタン
    if st.button("回答を評価"):
        if not api_key or api_key == "your_api_key":
            st.error("有効なOpenAI APIキーを入力してください。")
        elif not uploaded_file:
            st.error("CSVファイルをアップロードしてください。")
        else:
            try:
                with st.spinner("回答を評価しています..."):
                    # CSVファイルの内容を読み込む
                    csv_content = uploaded_file.read().decode('utf-8')

                    # リクエストボディの作成
                    payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "csv_content": csv_content
                    }

                    # バックエンドAPIのエンドポイント（ローカルホスト）
                    url = "http://localhost:8000/evaluate_answers"

                    # APIリクエストの送信
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        results = data["results"]
                        total_tokens = data["total_tokens"]
                        total_processing_time = data["total_processing_time"]

                        # 結果の表示
                        st.subheader("評価結果")
                        for result in results:
                            st.write(f"**Talk Nums**: {result['talk_nums']}")
                            st.write(f"**Task Name**: {result['task_name']}")
                            st.write(f"**Word**: {result['word']}")
                            st.write(f"**Query**: {result['query']}")
                            st.write(f"**Expected Answer**: {result['expected_answer']}")
                            st.write(f"**GPT Response**: {result['gpt_response']}")
                            st.write(f"**Is Correct**: {result['is_correct']}")
                            st.write(f"**Token Count**: {result['token_count']}")
                            st.write(f"**Processing Time**: {result['processing_time']:.2f} seconds")
                            st.write("---")

                        # DataFrameの作成
                        df = pd.DataFrame(results, columns=["talk_nums", "task_name", "word", "query", "expected_answer", "gpt_response", "is_correct", "token_count", "processing_time"])

                        # CSVのダウンロード
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="評価結果をCSVでダウンロード",
                            data=csv,
                            file_name=f"evaluation_results.csv",
                            mime='text/csv',
                        )
                    else:
                        st.error(f"エラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
