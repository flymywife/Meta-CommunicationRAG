import streamlit as st
import requests

def show_analysis_page():
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

    st.write("分析対象のタスクを選択してください。除外したいタスクのチェックを外せます。")

    # チェックボックス付きのタスクリストを表示
    selected_tasks = []
    checkboxes = {}

    st.write("### タスク一覧")
    for task_name in task_names:
        checkboxes[task_name] = st.checkbox(task_name, value=True)

    # 選択されたタスク名を取得
    selected_tasks = [task for task, checked in checkboxes.items() if checked]

    st.write("選択されたタスク:")
    st.write(selected_tasks)
