# vectorization_page.py

import streamlit as st
import requests

def show_analysis_page():
    st.title("分析")

    # タスク名の入力
    task_name = st.text_input("タスク名を入力してください:")


