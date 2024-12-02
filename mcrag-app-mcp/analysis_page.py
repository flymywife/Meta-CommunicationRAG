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
        ("クロス集計", "主成分分析", "ドリフトベクトル分析")
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



    elif analysis_method == "主成分分析":
        st.write("分析対象のタスクを選択してください。")
        selected_tasks = []
        checkboxes = {}

        st.write("### タスク一覧")
        for task_name in task_names:
            checkboxes[task_name] = st.checkbox(task_name, value=False)

        # 選択されたタスク名を取得
        selected_tasks = [task for task, checked in checkboxes.items() if checked]

        if selected_tasks:
            # 選択されたタスク名をバックエンドに送信してデータを取得
            backend_url = "http://localhost:8000/perform_pca"
            try:
                # リクエストボディの作成
                payload = {
                    "api_key": api_key,
                    "task_names": selected_tasks
                }
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    pca_data = data.get("data", [])
                    if not pca_data:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        df = pd.DataFrame(pca_data)

                        # グラフの描画
                        st.write("### 主成分分析の結果")
                        fig = px.scatter(
                            df,
                            x='PC1',
                            y='PC2',
                            color='task_name',
                            hover_data=['word'],
                            title='主成分分析（PCA）結果'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"主成分分析データの取得中にエラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"主成分分析データの取得中に例外が発生しました: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")


    if analysis_method == "ドリフトベクトル分析":
        st.write("分析対象のタスクを選択してください。チェックを入れたタスクが分析に含まれます。")
        selected_tasks = []
        checkboxes = {}

        st.write("### タスク一覧")
        for task_name in task_names:
            checkboxes[task_name] = st.checkbox(task_name, value=False)

        selected_tasks = [task for task, checked in checkboxes.items() if checked]

        if selected_tasks:
            # バックエンドにリクエストを送信
            backend_url = "http://localhost:8000/calculate_drift_direction"
            try:
                payload = {
                    "api_key": api_key,
                    "task_names": selected_tasks
                }
                response = requests.post(backend_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    drift_data = data.get("data", [])
                    if not drift_data:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        drift_df = pd.DataFrame(drift_data)
                        st.write("### ドリフトベクトルの解析結果")
                        st.dataframe(drift_df)

                        # コサイン類似度のヒストグラムを表示
                        fig = px.histogram(drift_df, x='cosine_similarity', nbins=50, title='ドリフトベクトルの分布')
                        st.plotly_chart(fig, use_container_width=True)

                        # 単語別の散布図を表示
                        fig_scatter = px.scatter(
                            drift_df,
                            x='word',
                            y='cosine_similarity',
                            color='task_name',
                            title='トピック別ドリフトベクトル'
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.error(f"ドリフトベクトルの解析中にエラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"ドリフトベクトルの解析中に例外が発生しました: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")

    