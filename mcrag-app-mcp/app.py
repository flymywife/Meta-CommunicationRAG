# app.py
import streamlit as st
from home import show_home
from conversation_page import show_conversation_page
from question_page import show_question_page
from evaluation_page import show_evaluation_page
from vectorization_page import show_vectorization_page
from analysis_page import show_analysis_page
from workflow_execution_page import show_workflow_execution_page
import os
from dotenv import load_dotenv

st.set_page_config(page_title="MCRAG", layout="wide")

# 環境変数のロード
load_dotenv()
default_api_key = os.getenv("OPENAI_API_KEY", "your_api_key")

# サイドバーの設定
st.sidebar.title("設定")

# OpenAI APIキーの入力欄
api_key = st.sidebar.text_input("OpenAI APIキーを入力してください:", value=default_api_key)

# Temperatureのスライダー
temperature = st.sidebar.slider("Temperatureを選択してください:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# サイドバーのメニュー
st.sidebar.title("メニュー")
page = st.sidebar.radio("ページを選択してください:", ('ホーム', '会話生成', '質問と回答の生成', "回答の評価", "ベクトル化", "ワークフロー実行","分析"))

# ページの表示
if page == 'ホーム':
    show_home()
elif page == '会話生成':
    show_conversation_page(api_key, temperature)
elif page == "質問と回答の生成":
    show_question_page(api_key, temperature)
elif page == "回答の評価":
    show_evaluation_page(api_key, temperature)
elif page == "ベクトル化":
    show_vectorization_page(api_key)
elif page == "ワークフロー実行":
    show_workflow_execution_page(api_key, temperature)
elif page == "分析":
    show_analysis_page()