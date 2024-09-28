import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import json

def show_question_page():
    # 環境変数のロード
    load_dotenv()
    default_api_key = os.getenv("OPENAI_API_KEY", "your_api_key")

    st.title("質問と回答の生成")

    # サイドバーの設定
    st.sidebar.title("設定")

    # OpenAI APIキーの入力欄
    api_key = st.sidebar.text_input("OpenAI APIキーを入力してください:", value=default_api_key)

    # Temperatureのスライダー
    temperature = st.sidebar.slider("Temperatureを選択してください:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # CSVファイルのアップロード
    uploaded_files = st.file_uploader("会話履歴のCSVファイルをアップロードしてください（複数選択可）:", type="csv", accept_multiple_files=True)

    # 質問と回答の生成ボタン
    if st.button("質問と回答を生成"):
        if not api_key or api_key == "your_api_key":
            st.error("有効なOpenAI APIキーを入力してください。")
        elif not uploaded_files:
            st.error("少なくとも1つのCSVファイルをアップロードしてください。")
        else:
            try:
                with st.spinner("質問と回答を生成しています..."):
                    # CSVファイルの内容を読み込む
                    csv_contents = []
                    for uploaded_file in uploaded_files:
                        csv_content = uploaded_file.read().decode('utf-8')
                        csv_contents.append(csv_content)

                    # リクエストボディの作成
                    payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "csv_contents": csv_contents
                    }

                    # バックエンドAPIのエンドポイント（ローカルホスト）
                    url = "http://localhost:8000/generate_questions"

                    # APIリクエストの送信
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        results = data["results"]
                        total_tokens = data["total_tokens"]
                        total_processing_time = data["total_processing_time"]

                        # 結果の表示
                        st.subheader("生成された質問と回答")
                        for result in results:
                            st.write(f"**Talk Nums**: {result['talk_nums']}")
                            st.write(f"**Task Name**: {result['task_name']}")
                            st.write(f"**Word**: {result['word']}")
                            st.write(f"**Question**: {result['query']}")
                            st.write(f"**Answer**: {result['answer']}")
                            st.write(f"**Token Count**: {result['token_count']}")
                            st.write(f"**Processing Time**: {result['processing_time']:.2f} seconds")
                            st.write("---")

                        # DataFrameの作成
                        df = pd.DataFrame(results, columns=["talk_nums", "task_name", "word", "query", "answer", "token_count", "processing_time"])

                        # CSVのダウンロード
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="CSVをダウンロード",
                            data=csv,
                            file_name=f"questions_and_answers.csv",
                            mime='text/csv',
                        )
                    else:
                        st.error(f"エラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
