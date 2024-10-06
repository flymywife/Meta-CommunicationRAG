Meta-Communication Retrieval-Augmented Generation(MCRAG)
このアプリは、1万文字以内の小説をアップロードし、テキストをチャンク分割して「いつ」「どこで」「誰で」の問いかけを作成するツールです。アプリは必ず正答します。

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

## アプリの説明

- アップロードされたテキストファイルをチャンク分割し、指定された問いかけに対応する部分を抽出します。  
- アプリは、「いつ」、「どこで」、「誰で」の問いかけに必ず正答します。

## 環境

- Python仮想環境を使用することを推奨します。

## 注意事項

- `.env` ファイルにAPIキーを必ず記載してください。



conda create --name mcrag-app python=3.10
conda activate mcrag-app



