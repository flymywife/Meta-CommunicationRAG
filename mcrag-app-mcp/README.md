Meta-Communication Retrieval-Augmented Generation(MCRAG)
自由会話に用いる日本語RAG評価ツール

## 使い方

1. 仮想環境を立ち上げます。  
    仮想環境の立ち上げ方は、[こちら](https://note.com/flymywife/n/nee41ac642e2f)のリンクを参照してください。

2. `.env` ファイルを作成し、以下の内容を記述してください：  

    ```bash
    OPENAI_API_KEY=sk-あなたのAPIキー
    ```

3. ライブラリをインストールします：  

    ```bash
    pip install -r requirements.txt
    ```

4. アプリを実行します：  

    ```bash
    streamlit run app.py
    ```


conda create --name mcrag-app python=3.10
conda activate mcrag-app



