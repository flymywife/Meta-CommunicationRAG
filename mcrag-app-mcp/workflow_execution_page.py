# workflow_execution_page.py

import streamlit as st
import pandas as pd
import requests
import json
import os

def show_workflow_execution_page(api_key, temperature):
    st.title("ワークフロー実行")

    # タスク名の入力
    task_name = st.text_input("タスク名を入力してください:")

    # ワードと情報の入力方法の選択
    input_method = st.radio("ワードと情報の入力方法を選択してください:", ('フォームで入力', 'CSVファイルをアップロード'))

    max_words = 20  # ワードの最大数
    max_infos = 5   # 情報の最大数

    words_info = []

    if input_method == 'フォームで入力':
        key_prefix = 'workflow_form'
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
        key_prefix = 'workflow_csv'
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

    # キャラクターのプロンプトの入力方法の選択
    character_prompt_input_method = st.radio("キャラクターのプロンプトの入力方法を選択してください:", ('フォームで入力', 'テキストファイルをアップロード'), key='workflow_character_prompt_method')

    if character_prompt_input_method == 'フォームで入力':
        character_prompt = st.text_area("キャラクターのプロンプトを入力してください:", height=150)
    else:
        character_prompt_file = st.file_uploader("キャラクターのプロンプトが記載されたテキストファイルをアップロードしてください:", type="txt", key='workflow_character_prompt_file')
        if character_prompt_file is not None:
            character_prompt = character_prompt_file.read().decode('utf-8')
        else:
            character_prompt = ''
        character_prompt = st.text_area("キャラクターのプロンプトを編集してください:", value=character_prompt, height=150)

    # ユーザー設定のプロンプトの入力方法の選択
    user_prompt_input_method = st.radio("ユーザー設定のプロンプトの入力方法を選択してください:", ('フォームで入力', 'テキストファイルをアップロード'), key='workflow_user_prompt_method')

    if user_prompt_input_method == 'フォームで入力':
        user_prompt = st.text_area("ユーザー設定のプロンプトを入力してください:", height=150)
    else:
        user_prompt_file = st.file_uploader("ユーザー設定のプロンプトが記載されたテキストファイルをアップロードしてください:", type="txt", key='workflow_user_prompt_file')
        if user_prompt_file is not None:
            user_prompt = user_prompt_file.read().decode('utf-8')
        else:
            user_prompt = ''
        user_prompt = st.text_area("ユーザー設定のプロンプトを編集してください:", value=user_prompt, height=150)

    # ワークフロー実行ボタン
    if st.button("ワークフローを実行"):
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
            # ワークフローの実行
            try:
                with st.spinner("ワークフローを実行しています..."):
                    # 進捗状況を表示するためのプレースホルダー
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    total_steps = 4
                    current_step = 0

                    # 1. 会話生成
                    progress_text.text("会話を生成しています...")
                    conversation_payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "task_name": task_name,
                        "words_info": words_info,
                        "character_prompt": character_prompt,
                        "user_prompt": user_prompt
                    }
                    conversation_url = "http://localhost:8000/generate_conversation"
                    conversation_response = requests.post(conversation_url, json=conversation_payload)

                    if conversation_response.status_code != 200:
                        st.error(f"会話生成中にエラーが発生しました: {conversation_response.text}")
                        return

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                    # 2. 質問と回答の生成
                    progress_text.text("質問と回答を生成しています...")
                    qa_payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "task_name": task_name
                    }
                    qa_url = "http://localhost:8000/generate_questions" 
                    qa_response = requests.post(qa_url, json=qa_payload)

                    if qa_response.status_code != 200:
                        st.error(f"質問と回答の生成中にエラーが発生しました: {qa_response.text}")
                        return

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                    # 3. ベクトル化
                    progress_text.text("ベクトル化しています...")
                    vector_payload = {
                        "api_key": api_key,
                        "task_name": task_name
                    }
                    vector_url = "http://localhost:8000/vectorize_conversations" 
                    vector_response = requests.post(vector_url, json=vector_payload)

                    if vector_response.status_code != 200:
                        st.error(f"ベクトル化中にエラーが発生しました: {vector_response.text}")
                        return

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                    # 4. 回答の評価
                    progress_text.text("回答を評価しています...")
                    evaluation_payload = {
                        "temperature": temperature,
                        "api_key": api_key,
                        "task_name": task_name
                    }
                    evaluation_url = "http://localhost:8000/evaluate_answers"
                    evaluation_response = requests.post(evaluation_url, json=evaluation_payload)

                    if evaluation_response.status_code != 200:
                        st.error(f"回答の評価中にエラーが発生しました: {evaluation_response.text}")
                        return

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                    # ワークフロー完了
                    progress_text.text("ワークフローが完了しました。")
                    st.success("すべてのステップが正常に完了しました。")

                    # 必要に応じて結果を表示
                    # ここでは、評価結果を表示します
                    evaluation_data = evaluation_response.json()
                    results = evaluation_data.get("results", [])
                    st.subheader("評価結果")
                    for result in results:
                        st.write(f"**Talk Nums**: {result['talk_nums']}")
                        st.write(f"**Task Name**: {result['task_name']}")
                        st.write(f"**Word**: {result['word']}")
                        st.write(f"**Query**: {result['query']}")
                        st.write(f"**Expected Answer**: {result['expected_answer']}")
                        st.write(f"**GPT Response**: {result['gpt_response']}")
                        st.write(f"**Is Correct**: {result['is_correct']}")
                        st.write(f"**Evaluation Detail**: {result['evaluation_detail']}")
                        st.write(f"**Model**: {result.get('model', '')}")
                        st.write("---")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
