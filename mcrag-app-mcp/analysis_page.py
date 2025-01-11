import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np

def show_analysis_page(api_key: str):
    """
    主成分分析や特異値分解など、分析処理を行うStreamlitページ。
    バックエンド（FastAPI）と通信し、結果を可視化します。
    """

    st.title("分析ページ")

    # ---------------------------
    # 1) タスク一覧をバックエンドから取得
    # ---------------------------
    backend_url = "http://localhost:8000"  # 必要に応じて変更してください
    try:
        resp = requests.get(f"{backend_url}/get_task_names")
        if resp.status_code == 200:
            data = resp.json()
            task_names = data.get("task_names", [])
        else:
            st.error(f"タスク名の取得中にエラー: {resp.text}")
            return
    except Exception as e:
        st.error(f"タスク名の取得中に例外が発生しました: {e}")
        return

    if not task_names:
        st.warning("データベースにタスクが存在しません。")
        return

    # ---------------------------
    # 2) 分析方法の選択
    # ---------------------------
    analysis_method = st.radio(
        "分析方法を選択してください：",
        ("クロス集計", "主成分分析", "主成分分析（トータル）", "特異値分解（SVD）")
    )

    # ------------------------------------------------
    # 3) 「クロス集計」の処理
    # ------------------------------------------------
    if analysis_method == "クロス集計":
        st.write("### クロス集計の対象タスク選択")
        checkboxes = {}
        for tname in task_names:
            checkboxes[tname] = st.checkbox(tname, value=False)

        selected_tasks = [t for t, checked in checkboxes.items() if checked]

        if selected_tasks:
            cross_tab_url = f"{backend_url}/get_cross_tab_data"
            try:
                payload = {"api_key": api_key, "task_names": selected_tasks}
                resp = requests.post(cross_tab_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    cross_tab_data = data.get("data", [])
                    if not cross_tab_data:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        df = pd.DataFrame(cross_tab_data)
                        st.write("### クロス集計結果")
                        st.dataframe(df)
                else:
                    st.error(f"クロス集計データ取得エラー: {resp.text}")
            except Exception as e:
                st.error(f"クロス集計データ取得中に例外: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")

    # ------------------------------------------------
    # 4) 「主成分分析」(既存：個別プロット)
    # ------------------------------------------------
    elif analysis_method == "主成分分析":
        st.write("### PCA（各タスク単位でプロット）")
        checkboxes = {}
        for tname in task_names:
            checkboxes[tname] = st.checkbox(tname, value=False)

        selected_tasks = [t for t, c in checkboxes.items() if c]
        if selected_tasks:
            pca_url = f"{backend_url}/perform_pca"
            try:
                payload = {"api_key": api_key, "task_names": selected_tasks}
                resp = requests.post(pca_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    pca_results = data.get("data", [])
                    if not pca_results:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        st.write("### 主成分分析（PCA）結果")

                        df = pd.DataFrame(pca_results)
                        # 'word' ごとにグループ化して可視化
                        grouped = df.groupby('word')
                        for word, group_df in grouped:
                            st.write(f"#### Word: {word}")

                            fig = px.scatter(
                                group_df,
                                x='PC1',
                                y='PC2',
                                color='Label',
                                hover_data=['qa_id', 'task_name', 'answer_type', 'model', 'distance'],
                                title=f'Word: {word}'
                            )

                            # 期待値("期待値")と"モデル: X"の点を線で結ぶ
                            for qa_id_val, qa_sub in group_df.groupby('qa_id'):
                                expected_row = qa_sub[qa_sub['answer_type'] == '期待値']
                                model_rows = qa_sub[qa_sub['answer_type'].str.startswith('モデル:')]
                                if len(expected_row) == 1:
                                    exp_x = expected_row['PC1'].values[0]
                                    exp_y = expected_row['PC2'].values[0]

                                    for _, mrow in model_rows.iterrows():
                                        model_x = mrow['PC1']
                                        model_y = mrow['PC2']
                                        distance = mrow.get('distance', None)
                                        hover_text = f"Distance: {distance:.4f}" if distance else "Distance: N/A"

                                        # Plotlyの図形機能で線を追加
                                        fig.add_shape(
                                            type="line",
                                            x0=exp_x, y0=exp_y,
                                            x1=model_x, y1=model_y,
                                            line=dict(color="gray", dash="dot")
                                        )
                            st.plotly_chart(fig, use_container_width=True)

                            # QA詳細を表示
                            qa_ids = group_df['qa_id'].unique()
                            for qid in qa_ids:
                                qa_data = group_df[group_df['qa_id'] == qid]
                                question_text = qa_data.iloc[0]['question_text']
                                expected_answer = qa_data.iloc[0]['expected_answer']
                                st.write(f"**QA ID: {qid}**")
                                st.write(f"**質問文:** {question_text}")
                                st.write(f"**期待値（模範回答）:** {expected_answer}")

                                # モデル回答を表示
                                models = qa_data['model'].unique()
                                for model_val in models:
                                    # 期待値はスキップ
                                    if model_val == 'expected':
                                        continue

                                    row_m = qa_data[
                                        (qa_data['model'] == model_val) &
                                        (qa_data['answer_type'].str.startswith('モデル:'))
                                    ]
                                    if not row_m.empty:
                                        model_answer = row_m.iloc[0]['model_answer']
                                        distance_val = row_m.iloc[0]['distance']
                                        st.write(f"- **モデル [{model_val}] の回答:** {model_answer}")
                                        st.write(f"  - 期待値からの距離: {distance_val}")
                                st.write("---")

                else:
                    st.error(f"バックエンドからPCA取得時にエラー: {resp.text}")
            except Exception as e:
                st.error(f"PCA通信中に例外: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")

    # ------------------------------------------------
    # 5) ★ 新規追加 ★ 「主成分分析（トータル）」
    #     期待値の行も表に記載し、distance=0.0 を表示する
    # ------------------------------------------------
    elif analysis_method == "主成分分析（トータル）":
        st.write("### PCA（トータル）: タスクをまとめて平均座標を可視化")
        checkboxes = {}
        for tname in task_names:
            checkboxes[tname] = st.checkbox(tname, value=False)

        selected_tasks = [t for t, c in checkboxes.items() if c]
        if selected_tasks:
            pca_url = f"{backend_url}/perform_pca"
            try:
                payload = {"api_key": api_key, "task_names": selected_tasks}
                resp = requests.post(pca_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    pca_results = data.get("data", [])
                    if not pca_results:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        st.write("### 主成分分析（トータル）の結果")
                        df = pd.DataFrame(pca_results)

                        # -------------------------
                        # (A) answer_typeごとに PC1, PC2 の平均を計算
                        # -------------------------
                        grouped_df = df.groupby('answer_type', as_index=False).agg({
                            'PC1': 'mean',
                            'PC2': 'mean'
                        })
                        # 例: answer_type が「期待値」 or 「モデル: GPT-4」など

                        fig = px.scatter(
                            grouped_df,
                            x='PC1',
                            y='PC2',
                            color='answer_type',
                            text='answer_type',
                            title='主成分分析（トータル）: 回答タイプ別 平均座標'
                        )
                        fig.update_traces(textposition='top center')

                        # -------------------------
                        # (B) 期待値平均点 & 各モデル平均点の距離を計算
                        #     期待値行も含めて表に載せる
                        # -------------------------
                        expected_row = grouped_df[grouped_df['answer_type'] == '期待値']
                        if len(expected_row) == 1:
                            exp_pc1 = expected_row.iloc[0]['PC1']
                            exp_pc2 = expected_row.iloc[0]['PC2']

                            # 期待値 + 各モデル行をまとめる
                            model_rows = grouped_df.copy()

                            # distance_from_expected を計算
                            distances = []
                            for _, row_m in model_rows.iterrows():
                                if row_m['answer_type'] == '期待値':
                                    dist_val = 0.0  # 自分自身なので距離0
                                else:
                                    dist_val = np.sqrt((exp_pc1 - row_m['PC1'])**2 + (exp_pc2 - row_m['PC2'])**2)
                                distances.append(dist_val)

                            model_rows['distance_from_expected'] = distances

                            st.write("#### 期待値含む一覧 (PC1, PC2, distance)")
                            st.dataframe(model_rows[['answer_type', 'PC1', 'PC2', 'distance_from_expected']])

                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"バックエンドからPCA取得時にエラー: {resp.text}")
            except Exception as e:
                st.error(f"PCA通信中に例外: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")

    # ------------------------------------------------
    # 6) 「特異値分解（SVD）」の処理
    # ------------------------------------------------
    elif analysis_method == "特異値分解（SVD）":
        st.write("### SVD（特異値分解）")
        checkboxes = {}
        for tname in task_names:
            checkboxes[tname] = st.checkbox(tname, value=False)

        selected_tasks = [t for t, c in checkboxes.items() if c]
        if selected_tasks:
            svd_url = f"{backend_url}/perform_svd_analysis"
            try:
                payload = {"api_key": api_key, "task_names": selected_tasks}
                resp = requests.post(svd_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    svd_results = data.get("data", [])
                    if not svd_results:
                        st.warning("選択されたタスクに対応するデータが見つかりませんでした。")
                    else:
                        st.write("### 特異値分解（SVD）の結果")
                        for item in svd_results:
                            qa_id = item.get('qa_id')
                            question_text = item.get('question_text')
                            expected_answer = item.get('expected_answer')
                            model_answer = item.get('model_answer')
                            coords = item.get('coordinates', [])

                            df_coords = pd.DataFrame(coords)
                            fig = px.scatter(
                                df_coords,
                                x='expected',
                                y='actual',
                                color='Label',
                                title=f'QA ID: {qa_id}, 質問: {question_text}'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # 表示
                            st.write("**質問文:**", question_text)
                            st.write("**期待値（模範回答）:**", expected_answer)
                            st.write("**モデル回答:**", model_answer)
                            st.write("---")

                else:
                    st.error(f"バックエンドからSVDデータ取得時にエラー: {resp.text}")
            except Exception as e:
                st.error(f"SVD通信中に例外: {e}")
        else:
            st.info("少なくとも1つのタスクを選択してください。")


# -----------------------------------------------
# 単体でこのファイルを streamlit run する場合
# -----------------------------------------------
if __name__ == "__main__":
    st.sidebar.title("APIキー入力")
    user_api_key = st.sidebar.text_input("API Key", value="", type="password")
    if user_api_key:
        show_analysis_page(api_key=user_api_key)
    else:
        st.info("左側のサイドバーにAPIキーを入力してください。")
