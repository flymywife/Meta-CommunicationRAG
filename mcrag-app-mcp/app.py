import streamlit as st
from home import show_home
from conversation_page import show_conversation_page
from question_page import show_question_page
from evaluation_page import show_evaluation_page


st.set_page_config(page_title="AITuber 会話生成ツール", layout="wide")

# サイドバーのメニュー
st.sidebar.title("メニュー")
page = st.sidebar.radio("ページを選択してください:", ('ホーム', '会話生成', '質問と回答の生成', "回答の評価"))

# ページの表示
if page == 'ホーム':
    show_home()
elif page == '会話生成':
    show_conversation_page()
elif page == "質問と回答の生成":
    show_question_page()
elif page == "回答の評価":
    show_evaluation_page()
