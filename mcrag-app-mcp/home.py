# home.py

import streamlit as st
import os

def show_home():
    st.title("Meta-Communication Retrieval-Augmented Generation(MCRAG)")
    st.header("サンプルとマニュアルのダウンロード")
    st.write("サンプルとマニュアルのzipです。MCRAGの機能と使い方と、インプットデータのフォーマットとサンプルが置いています。")

    # サンプルファイルのパスを設定
    sample_file_path = os.path.join("sample", "mcrag_sample.zip")

    # サンプルファイルが存在するか確認
    if os.path.exists(sample_file_path):
        # ファイルをバイナリモードで読み込む
        with open(sample_file_path, "rb") as file:
            file_bytes = file.read()
            st.download_button(
                label="サンプルとマニュアルダウンロード",
                data=file_bytes,
                file_name="mcrag_sample.zip",
                mime="application/zip"
            )
    else:
        st.error("サンプルファイルが見つかりません。ファイルの場所を確認してください。")
