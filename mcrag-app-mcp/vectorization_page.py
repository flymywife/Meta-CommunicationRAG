# vectorization_page.py

import streamlit as st
import os
from dotenv import load_dotenv
import requests

def show_vectorization_page():
    # 環境変数のロード
    load_dotenv()
    default_api_key = os.getenv("OPENAI_API_KEY", "your_api_key")

    st.title("ベクトル化")

    # サイドバーの設定
    st.sidebar.title("設定")

    # OpenAI APIキーの入力欄
    api_key = st.sidebar.text_input("OpenAI APIキーを入力してください:", value=default_api_key)

    # タスク名の入力
    task_name = st.text_input("タスク名を入力してください:")

    # ベクトル化ボタン
    if st.button("ベクトル化を実行"):
        if not api_key or api_key == "your_api_key":
            st.error("有効なOpenAI APIキーを入力してください。")
        elif not task_name:
            st.error("タスク名を入力してください。")
        else:
            try:
                with st.spinner("ベクトル化を実行しています..."):
                    # リクエストボディの作成
                    payload = {
                        "api_key": api_key,
                        "task_name": task_name
                    }

                    # バックエンドAPIのエンドポイント（ローカルホスト）
                    url = "http://localhost:8000/vectorize_conversations"

                    # APIリクエストの送信
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["message"])
                    else:
                        # エラーメッセージを表示
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            error_detail = response.text
                        st.error(f"エラーが発生しました: {error_detail}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
