# question_page.py

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
    temperature = st.sidebar.slider("Temperatureを選択してください:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    # JSONファイルのアップロード
    uploaded_file = st.file_uploader("会話データのJSONファイルをアップロードしてください:", type="json")

    if uploaded_file is not None:
        # JSONファイルの内容を読み込む
        try:
            json_data = json.load(uploaded_file)
            st.success("JSONファイルが正常に読み込まれました。")
            # ファイル名を取得
            json_file_name = uploaded_file.name
        except Exception as e:
            st.error(f"JSONファイルの読み込み中にエラーが発生しました: {e}")
            return

        # 質問と回答の生成ボタン
        if st.button("質問と回答を生成"):
            if not api_key or api_key == "your_api_key":
                st.error("有効なOpenAI APIキーを入力してください。")
            else:
                try:
                    with st.spinner("質問と回答を生成しています..."):
                        # リクエストボディの作成
                        payload = {
                            "temperature": temperature,
                            "api_key": api_key,
                            "json_data": json_data
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
                                st.write("---")

                            # 合計のトークン数と処理時間を表示
                            st.write(f"**Total Token Count**: {total_tokens}")
                            st.write(f"**Total Processing Time**: {total_processing_time:.2f} seconds")

                            # 結果をJSONでダウンロード
                            results_json = json.dumps(results, ensure_ascii=False, indent=2).encode('utf-8')
                            st.download_button(
                                label="結果をJSONでダウンロード",
                                data=results_json,
                                file_name=f"questions_{json_file_name}",
                                mime='application/json',
                            )
                        else:
                            st.error(f"エラーが発生しました: {response.text}")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
    else:
        st.info("会話データのJSONファイルをアップロードしてください。")
