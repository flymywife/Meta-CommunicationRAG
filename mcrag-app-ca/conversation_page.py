# conversation_page.py

import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import json

def show_conversation_page():
    # 環境変数のロード
    load_dotenv()
    default_api_key = os.getenv("OPENAI_API_KEY", "your_api_key")

    st.title("会話生成")

    # サイドバーの設定
    st.sidebar.title("設定")

    # OpenAI APIキーの入力欄
    api_key = st.sidebar.text_input("OpenAI APIキーを入力してください:", value=default_api_key)

    # タスク名の入力
    task_name = st.text_input("タスク名を入力してください:")

    # ワードと情報の入力方法の選択
    input_method = st.radio("ワードと情報の入力方法を選択してください:", ('フォームで入力', 'CSVファイルをアップロード'))

    max_words = 20  # ワードの最大数

    words_info = {}

    if input_method == 'フォームで入力':
        key_prefix = 'form'
        # デフォルトのワード数
        num_words = st.number_input("ワードの数を選択してください:", min_value=1, max_value=max_words, value=1, step=1)

        # ワードと情報の入力フィールドを動的に生成
        for i in range(int(num_words)):
            st.write(f"### ワード {i+1}")
            word_input = st.text_input(f"ワード {i+1} を入力してください:", key=f"{key_prefix}_word_{i}")
            info_input = st.text_area(f"ワード '{word_input}' の情報を入力してください:", height=100, key=f"{key_prefix}_info_{i}")
            if word_input:
                # カンマでワードを分割
                words = [w.strip() for w in word_input.split(',') if w.strip()]
                for word in words:
                    words_info[word] = info_input

    elif input_method == 'CSVファイルをアップロード':
        key_prefix = 'csv'
        # CSVファイルのアップロード
        uploaded_file = st.file_uploader("ワードと情報が記載されたCSVファイルをアップロードしてください（ヘッダー行が必要です）:", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # ヘッダーの数を20個に制限
            df = df.iloc[:, :2]  # 最初の2列のみ（ワードと情報）
            df.columns = ['word', 'info']
            df = df.head(max_words)  # 最初の20行のみ
            words_info = dict(zip(df['word'], df['info']))

            # フォームに反映
            num_words = len(words_info)
            st.success(f"{num_words} 個のワードが読み込まれました。必要に応じて編集してください。")
            for i, (word, info) in enumerate(words_info.items()):
                st.write(f"### ワード {i+1}")
                word_input = st.text_input(f"ワード {i+1} を入力してください:", value=word, key=f"{key_prefix}_word_{i}")
                info_input = st.text_area(f"ワード '{word_input}' の情報を入力してください:", value=info, height=100, key=f"{key_prefix}_info_{i}")
                if word_input:
                    # カンマでワードを分割
                    words = [w.strip() for w in word_input.split(',') if w.strip()]
                    for word in words:
                        words_info[word] = info_input

    # AITuberのプロンプト入力
    aituber_prompt = st.text_area("AITuberのプロンプトを入力してください:", height=150)

    # ユーザー設定のプロンプト入力
    user_prompt = st.text_area("ユーザー設定のプロンプトを入力してください:", height=150)

    # Temperatureのスライダー
    temperature = st.slider("Temperatureを選択してください:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    # ワードの全ての情報が含まれるまでの会話数を1-5まで指定
    num_turns_per_word = st.slider("ワードの情報が全て含まれるまでの会話数を選択してください (1-5):", min_value=1, max_value=5, value=1, step=1)

    # 会話生成ボタン
    if st.button("会話生成"):
        if not task_name:
            st.error("タスク名を入力してください。")
        elif not api_key or api_key == "your_api_key":
            st.error("有効なOpenAI APIキーを入力してください。")
        else:
            # バックエンドAPIにリクエストを送信
            try:
                with st.spinner("会話を生成しています..."):
                    # リクエストボディの作成
                    payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "task_name": task_name,
                        "words_info": words_info,
                        "num_turns_per_word": num_turns_per_word,
                        "aituber_prompt": aituber_prompt,
                        "user_prompt": user_prompt
                    }

                    # バックエンドAPIのエンドポイント（ローカルホスト）
                    url = "http://localhost:8000/generate_conversation"

                    # APIリクエストの送信
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        conversations = data["conversations"]
                        total_tokens = data["total_tokens"]
                        total_processing_time = data["total_processing_time"]
                        # 結果を表示
                        for turn in conversations:
                            st.write(f"**Talk {turn['talk_num']}**")
                            st.write(f"**Word**: {turn['word']}")
                            st.write(f"**User**: {turn['user']}")
                            st.write(f"**Assistant**: {turn['assistant']}")
                            st.write(f"**Token Count**: {turn['token_count']}")
                            st.write(f"**Processing Time**: {turn['processing_time']:.2f} seconds")
                            st.write("---")
                        # 合計のトークン数と処理時間を表示
                        st.write(f"**Total Token Count**: {total_tokens}")
                        st.write(f"**Total Processing Time**: {total_processing_time:.2f} seconds")

                        # DataFrameの作成
                        df = pd.DataFrame(conversations, columns=["talk_num", "task_name", "word", "user", "assistant", "token_count", "processing_time"])

                        # CSVのダウンロード
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="CSVをダウンロード",
                            data=csv,
                            file_name=f"{task_name}_conversation.csv",
                            mime='text/csv',
                        )
                    else:
                        st.error(f"エラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
