import streamlit as st
import pandas as pd
import requests
import json

def show_conversation_page(api_key, temperature):
    st.title("会話生成")

    # --- セッションステートの初期化 ---
    if "task_name" not in st.session_state:
        st.session_state["task_name"] = ""
    if "character_prompt" not in st.session_state:
        st.session_state["character_prompt"] = ""
    if "user_prompt" not in st.session_state:
        st.session_state["user_prompt"] = ""
    if "words_info" not in st.session_state:
        st.session_state["words_info"] = []  # ここでワード情報を保持する

    # --- タスク名の入力 ---
    st.session_state["task_name"] = st.text_input(
        "タスク名を入力してください:",
        value=st.session_state["task_name"],
        key="task_name_input"
    )

    # --- ワードと情報の入力方法の選択 ---
    input_method = st.radio(
        "ワードと情報の入力方法を選択してください:", 
        ('フォームで入力', 'CSVファイルをアップロード')
    )

    max_words = 20  # ワードの最大数
    max_infos = 5   # 情報の最大数

    # words_info を一時的に操作するリスト
    temp_words_info = []

    if input_method == 'フォームで入力':
        # --- フォーム入力 ---
        key_prefix = 'form'
        num_words = st.number_input(
            "ワードの数を選択してください:",
            min_value=1,
            max_value=max_words,
            value=1,
            step=1
        )

        # 一旦ローカルで集める
        for i in range(int(num_words)):
            st.write(f"### ワード {i+1}")
            word_input = st.text_input(
                f"ワード {i+1} を入力してください:",
                key=f"{key_prefix}_word_{i}"
            )
            infos = []
            for j in range(max_infos):
                info_input = st.text_area(
                    f"ワード '{word_input}' の情報 {j+1} を入力してください:",
                    height=80,
                    key=f"{key_prefix}_info_{i}_{j}"
                )
                infos.append(info_input.strip())
            if word_input:
                # カンマ区切りで複数ワードを入力した場合に対応
                words = [w.strip() for w in word_input.split(',') if w.strip()]
                for word in words:
                    temp_words_info.append({'word': word, 'infos': infos})

        # 入力結果をセッションステートに保存
        st.session_state["words_info"] = temp_words_info

    elif input_method == 'CSVファイルをアップロード':
        # --- CSV アップロード ---
        key_prefix = 'csv'
        uploaded_file = st.file_uploader(
            "ワードと情報が記載されたCSVファイルをアップロードしてください（ヘッダー行が必要です）:",
            type="csv"
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # ヘッダーの検証
            expected_columns = ['word'] + [f'info{i+1}' for i in range(max_infos)]
            if list(df.columns) != expected_columns:
                st.error(f"CSVのヘッダーが正しくありません。ヘッダーは {', '.join(expected_columns)} である必要があります。")
                return
            # 必要な列のみを取得し、最初の max_words 行のみ使用
            df = df.reindex(columns=expected_columns)
            df = df.head(max_words)
            # NaN を空文字列に変換
            df = df.fillna('')

            temp_words_info = []
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
                temp_words_info.append({'word': word, 'infos': infos})

            # CSV から読み込んだ内容をセッションステートに保存
            st.session_state["words_info"] = temp_words_info

            st.success(f"{len(temp_words_info)} 個のワードが読み込まれました。必要に応じて編集してください。")

            # --- 読み込んだあと、さらに編集フォームを出す (任意) ---
            updated_words_info = []
            for i, word_info in enumerate(temp_words_info):
                st.write(f"### ワード {i+1}")
                word_input = st.text_input(
                    f"ワード {i+1} を入力してください:",
                    value=word_info['word'],
                    key=f"{key_prefix}_word_{i}"
                )
                infos = []
                num_infos = len(word_info['infos'])
                for j in range(num_infos):
                    info_value = word_info['infos'][j]
                    info_input = st.text_area(
                        f"ワード '{word_input}' の情報 {j+1} を入力してください:",
                        value=info_value,
                        height=80,
                        key=f"{key_prefix}_info_{i}_{j}"
                    )
                    infos.append(info_input.strip())
                
                if word_input:
                    words = [w.strip() for w in word_input.split(',') if w.strip()]
                    for word in words:
                        updated_words_info.append({'word': word, 'infos': infos})

            # フォームの編集結果をセッションステートに保存
            st.session_state["words_info"] = updated_words_info

    # --- キャラクターのプロンプトの入力方法 ---
    character_prompt_input_method = st.radio(
        "キャラクターのプロンプトの入力方法を選択してください:",
        ('フォームで入力', 'テキストファイルをアップロード'),
        key='character_prompt_method'
    )

    if character_prompt_input_method == 'フォームで入力':
        st.session_state["character_prompt"] = st.text_area(
            "キャラクターのプロンプトを入力してください:",
            height=150,
            value=st.session_state["character_prompt"],
            key="character_prompt_input"
        )
    else:
        character_prompt_file = st.file_uploader(
            "キャラクターのプロンプトが記載されたテキストファイルをアップロードしてください:",
            type="txt",
            key='character_prompt_file'
        )
        if character_prompt_file is not None:
            file_content = character_prompt_file.read().decode('utf-8')
            st.session_state["character_prompt"] = file_content
        
        # エディタに反映
        st.session_state["character_prompt"] = st.text_area(
            "キャラクターのプロンプトを編集してください:",
            value=st.session_state["character_prompt"],
            height=150,
            key="character_prompt_input2"
        )

    # --- ユーザー設定のプロンプトの入力方法 ---
    user_prompt_input_method = st.radio(
        "ユーザー設定のプロンプトの入力方法を選択してください:",
        ('フォームで入力', 'テキストファイルをアップロード'),
        key='user_prompt_method'
    )

    if user_prompt_input_method == 'フォームで入力':
        st.session_state["user_prompt"] = st.text_area(
            "ユーザー設定のプロンプトを入力してください:",
            height=150,
            value=st.session_state["user_prompt"],
            key="user_prompt_input"
        )
    else:
        user_prompt_file = st.file_uploader(
            "ユーザー設定のプロンプトが記載されたテキストファイルをアップロードしてください:",
            type="txt",
            key='user_prompt_file'
        )
        if user_prompt_file is not None:
            file_content = user_prompt_file.read().decode('utf-8')
            st.session_state["user_prompt"] = file_content
        
        st.session_state["user_prompt"] = st.text_area(
            "ユーザー設定のプロンプトを編集してください:",
            value=st.session_state["user_prompt"],
            height=150,
            key="user_prompt_input2"
        )

    # --- 会話生成ボタン ---
    if st.button("会話生成"):
        # セッションステートから値を取り出してチェック
        task_name = st.session_state["task_name"]
        character_prompt = st.session_state["character_prompt"]
        user_prompt = st.session_state["user_prompt"]
        words_info = st.session_state["words_info"]

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
                    payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "task_name": task_name,
                        "words_info": words_info,
                        "character_prompt": character_prompt,
                        "user_prompt": user_prompt
                    }
                    url = "http://localhost:8000/generate_conversation"
                    response = requests.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        conversations = data["conversations"]
                        total_tokens = data["total_tokens"]
                        total_processing_time = data["total_processing_time"]

                        for turn in conversations:
                            st.write(f"**Talk {turn['talk_num']}**")
                            st.write(f"**Word**: {turn['word']}")
                            st.write(f"**Info**: {turn['info']}")
                            st.write(f"**User**: {turn['user']}")
                            st.write(f"**Assistant**: {turn['assistant']}")
                            st.write(f"**Processing Time**: {float(turn['processing_time']):.2f} seconds")
                            st.write("---")

                        st.write(f"**Total Token Count**: {total_tokens}")
                        st.write(f"**Total Processing Time**: {total_processing_time:.2f} seconds")

                        # JSONのダウンロードボタン
                        json_data = json.dumps({
                            "task_name": task_name,
                            "character_prompt": character_prompt,
                            "user_prompt": user_prompt,
                            "conversations": conversations
                        }, ensure_ascii=False, indent=2).encode('utf-8')

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
