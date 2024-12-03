import streamlit as st
import requests
import plotly.express as px  # Plotlyを使用してグラフを描画

import pandas as pd

def show_analysis_page(api_key):
    st.title("分析")

    # バックエンドからタスク名を取得
    backend_url = "http://localhost:8000/get_task_names"
    try:
        response = requests.get(backend_url)
        if response.status_code == 200:
            data = response.json()
            task_names = data.get("task_names", [])
        else:
            st.error(f"タスク名の取得中にエラーが発生しました: {response.text}")
            return
    except Exception as e:
        st.error(f"タスク名の取得中に例外が発生しました: {e}")
        return

    if not task_names:
        st.warning("データベースにタスクが存在しません。")
        return

    # 分析方法の選択
    analysis_method = st.radio(
        "分析方法を選択してください：",
        ("クロス集計", "主成分分析",  "特異値分解（SVD）")
    )


    if analysis_method == "クロス集計":
        st.write("分析対象のタスクを選択してください。チェックを入れたタスクが分析に含まれます。")

        # チェックボックス付きのタスクリストを表示（デフォルトでチェックなし）
        selected_tasks = []
        checkboxes = {}

        st.write("### タスク一覧")
        for task_name in task_names:
            checkboxes[task_name] = st.checkbox(task_name, value=False)

        # 選択されたタスク名を取得
        selected_tasks = [task for task, checked in checkboxes.items() if checked]

        if selected_tasks:
            # 選択されたタスク名をバックエンドに送信してデータを取得
            backend_url = "http://localhost:8000/get_cross_tab_data"
            try:
                # リクエストボディの作成
                payload = {
                    "api_key": api_key,
                    "task_names": selected_tasks
                }
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    cross_tab_data = data.get("data", [])
                    if not cross_tab_data:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        df = pd.DataFrame(cross_tab_data)
                        st.write("### クロス集計結果")
                        st.dataframe(df)
                else:
                    st.error(f"クロス集計データの取得中にエラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"クロス集計データの取得中に例外が発生しました: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")



    if analysis_method == "主成分分析":
        st.write("分析対象のタスクを選択してください。")
        selected_tasks = []
        checkboxes = {}

        st.write("### タスク一覧")
        for task_name in task_names:
            checkboxes[task_name] = st.checkbox(task_name, value=False)

        # 選択されたタスク名を取得
        selected_tasks = [task for task, checked in checkboxes.items() if checked]

        if selected_tasks:
            # バックエンドに選択されたタスク名を送信してPCA分析を実行
            backend_url = "http://localhost:8000/perform_pca"
            try:
                payload = {
                    "api_key": api_key,
                    "task_names": selected_tasks
                }
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    pca_results = data.get("data", [])
                    if not pca_results:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        st.write("### 主成分分析（PCA）の結果")

                        df = pd.DataFrame(pca_results)

                        # 'word' ごとにデータをグループ化
                        grouped = df.groupby('word')

                        for word, group_df in grouped:
                            st.write(f"#### Word: {word}")

                            # グラフの描画
                            fig = px.scatter(
                                group_df,
                                x='PC1',
                                y='PC2',
                                color='Label',
                                hover_data=['qa_id', 'task_name', 'answer_type', 'model'],
                                title=f'Word: {word}'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # 各 qa_id ごとに質問と回答を表示
                            qa_ids = group_df['qa_id'].unique()
                            for qa_id in qa_ids:
                                qa_data = group_df[group_df['qa_id'] == qa_id]
                                question_text = qa_data.iloc[0]['question_text']
                                expected_answer = qa_data.iloc[0]['expected_answer']
                                st.write(f"**QA ID: {qa_id}**")
                                st.write(f"**質問文:** {question_text}")
                                st.write(f"**期待値（模範回答）:** {expected_answer}")

                                # モデルごとの回答を表示
                                models = qa_data['model'].unique()
                                for model in models:
                                    model_data = qa_data[(qa_data['qa_id'] == qa_id) & (qa_data['model'] == model)]
                                    model_answer = model_data.iloc[0]['model_answer']
                                    st.write(f"**モデル [{model}] の回答:** {model_answer}")

                                st.write("---")

                else:
                    st.error(f"PCAデータの取得中にエラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"PCAデータの取得中に例外が発生しました: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")
    
    elif analysis_method == "特異値分解（SVD）":
        st.write("分析対象のタスクを選択してください。")

        # タスク選択のチェックボックス
        selected_tasks = []
        checkboxes = {}
        st.write("### タスク一覧")
        for task_name in task_names:
            checkboxes[task_name] = st.checkbox(task_name, value=False)

        selected_tasks = [task for task, checked in checkboxes.items() if checked]

        if selected_tasks:
            # バックエンドに選択されたタスク名を送信してSVD分析を実行
            backend_url = "http://localhost:8000/perform_svd_analysis"
            try:
                payload = {
                    "api_key": api_key,
                    "task_names": selected_tasks
                }
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    svd_results = data.get("data", [])
                    if not svd_results:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        st.write("### 特異値分解（SVD）の結果")

                        # 各結果についてループし、グラフとテキストを表示
                        for result in svd_results:
                            qa_id = result['qa_id']
                            question_text = result['question_text']
                            expected_answer = result['expected_answer']
                            model_answer = result['model_answer']
                            coordinates = result['coordinates']

                            # グラフ描画用のデータフレームを作成
                            df = pd.DataFrame(coordinates)

                            # グラフの描画
                            fig = px.scatter(
                                df,
                                x='expected',
                                y='actual',
                                color='Label',
                                title=f'QA ID: {qa_id}, 質問: {question_text}'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # 期待値とモデルの回答を表示
                            st.write("**期待値（模範回答）:**")
                            st.write(expected_answer)
                            st.write("**モデルの回答:**")
                            st.write(model_answer)
                            st.write("---")  # 区切り線

                else:
                    st.error(f"SVDデータの取得中にエラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"SVDデータの取得中に例外が発生しました: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")