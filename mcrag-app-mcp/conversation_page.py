# conversation_page.py
import streamlit as st
import pandas as pd
import requests
import json

def show_conversation_page(api_key, temperature):
    st.title("会話生成")

    # タスク名の入力
    task_name = st.text_input("タスク名を入力してください:")

    # ワードと情報の入力方法の選択
    input_method = st.radio("ワードと情報の入力方法を選択してください:", ('フォームで入力', 'CSVファイルをアップロード'))

    max_words = 20  # ワードの最大数
    max_infos = 5   # 情報の最大数

    words_info = []

    if input_method == 'フォームで入力':
        key_prefix = 'form'
        # デフォルトのワード数
        num_words = st.number_input("ワードの数を選択してください:", min_value=1, max_value=max_words, value=1, step=1)

        # ワードと情報の入力フィールドを動的に生成
        for i in range(int(num_words)):
            st.write(f"### ワード {i+1}")
            word_input = st.text_input(f"ワード {i+1} を入力してください:", key=f"{key_prefix}_word_{i}")
            infos = []
            for j in range(max_infos):
                info_input = st.text_area(f"ワード '{word_input}' の情報 {j+1} を入力してください:", height=100, key=f"{key_prefix}_info_{i}_{j}")
                infos.append(info_input.strip())
            if word_input:
                # カンマでワードを分割
                words = [w.strip() for w in word_input.split(',') if w.strip()]
                for word in words:
                    words_info.append({'word': word, 'infos': infos})

    elif input_method == 'CSVファイルをアップロード':
        key_prefix = 'csv'
        # CSVファイルのアップロード
        uploaded_file = st.file_uploader("ワードと情報が記載されたCSVファイルをアップロードしてください（ヘッダー行が必要です）:", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # ヘッダーの検証
            expected_columns = ['word'] + [f'info{i+1}' for i in range(max_infos)]
            if list(df.columns) != expected_columns:
                st.error(f"CSVのヘッダーが正しくありません。ヘッダーは {', '.join(expected_columns)} である必要があります。")
                return
            # 必要な列のみを取得
            df = df.reindex(columns=expected_columns)
            df = df.head(max_words)  # 最初の20行のみ
            # NaN を空文字列に変換
            df = df.fillna('')
            words_info = []
            for index, row in df.iterrows():
                word = str(row['word']).strip()
                if not word:
                    st.error(f"{index+1} 行目の 'word' が空です。'word' は必須です。")
                    return
                info1 = str(row['info1']).strip()
                if not info1:
                    st.error(f"{index+1} 行目の 'info1' が空です。'info1' は必須です。")
                    return
                infos = []
                for i in range(max_infos):
                    info_key = f'info{i+1}'
                    if info_key in row:
                        info = str(row[info_key]).strip()
                        if info:
                            infos.append(info)
                words_info.append({'word': word, 'infos': infos})

            # フォームに反映
            num_words = len(words_info)
            st.success(f"{num_words} 個のワードが読み込まれました。必要に応じて編集してください。")
            updated_words_info = []
            for i, word_info in enumerate(words_info):
                st.write(f"### ワード {i+1}")
                word_input = st.text_input(f"ワード {i+1} を入力してください:", value=word_info['word'], key=f"{key_prefix}_word_{i}")
                infos = []
                num_infos = len(word_info['infos'])
                for j in range(num_infos):
                    info_value = word_info['infos'][j]
                    info_input = st.text_area(f"ワード '{word_input}' の情報 {j+1} を入力してください:", value=info_value, height=100, key=f"{key_prefix}_info_{i}_{j}")
                    infos.append(info_input.strip())
                if word_input:
                    # カンマでワードを分割
                    words = [w.strip() for w in word_input.split(',') if w.strip()]
                    for word in words:
                        updated_words_info.append({'word': word, 'infos': infos})
            # ループが終了した後に words_info を更新
            words_info = updated_words_info

    # キャラクターのプロンプト入力
    character_prompt = st.text_area("キャラクターのプロンプトを入力してください:", height=150)

    # ユーザー設定のプロンプト入力
    user_prompt = st.text_area("ユーザー設定のプロンプトを入力してください:", height=150)

    # 会話生成ボタン
    if st.button("会話生成"):
        if not task_name:
            st.error("タスク名を入力してください。")
        elif not api_key or api_key == "your_api_key":
            st.error("有効なOpenAI APIキーを入力してください。")
        elif not character_prompt:
            st.error("キャラクターのプロンプトを入力してください。")
        elif not user_prompt:
            st.error("ユーザー設定のプロンプトを入力してください。")
        elif not words_info:
            st.error("少なくとも1つのワードと情報を入力してください。")
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
                        "character_prompt": character_prompt,
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
                            st.write(f"**Info**: {turn['info']}")
                            st.write(f"**User**: {turn['user']}")
                            st.write(f"**Assistant**: {turn['assistant']}")
                            st.write(f"**Token Count**: {turn['token_count']}")
                            st.write(f"**Processing Time**: {float(turn['processing_time']):.2f} seconds")
                            st.write("---")
                        # 合計のトークン数と処理時間を表示
                        st.write(f"**Total Token Count**: {total_tokens}")
                        st.write(f"**Total Processing Time**: {total_processing_time:.2f} seconds")

                        # JSONデータの作成
                        json_data = json.dumps({
                            "task_name": task_name,
                            "character_prompt": character_prompt,
                            "user_prompt": user_prompt,
                            "conversations": conversations
                        }, ensure_ascii=False, indent=2).encode('utf-8')
                        # JSONのダウンロード
                        st.download_button(
                            label="JSONをダウンロード",
                            data=json_data,
                            file_name=f"{task_name}_conversation.json",
                            mime='application/json',
                            )
                    else:
                        st.error(f"エラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
