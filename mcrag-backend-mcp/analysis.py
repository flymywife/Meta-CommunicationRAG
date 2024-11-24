# analysis.py

from database import ConversationDatabase
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
# vector_db.py をインポート
from vector_db import VectorDatabase

class Analysis:
    def __init__(self, api_key, db_file='conversation_data.db', index_folder='faiss_index',
                 index_file='faiss_index.index'):
        self.db = ConversationDatabase(db_file=db_file)
        self.vector_db = VectorDatabase(api_key=api_key, db_file=db_file, index_folder=index_folder, index_file=index_file)


    def get_all_task_names(self):
        try:
            select_sql = 'SELECT DISTINCT task_name FROM tasks'
            self.db.cursor.execute(select_sql)
            rows = self.db.cursor.fetchall()
            task_names = [row[0] for row in rows]
            return task_names
        except Exception as e:
            print(f"Error fetching task names: {e}")
            return []

    def get_cross_tab_data(self, selected_tasks):
        try:
            placeholders = ','.join('?' * len(selected_tasks))
            select_sql = f'''
            SELECT
                ea.model,
                ea.talk_nums,
                ea.word_info_id,
                ea.get_talk_nums
            FROM evaluated_answers ea
            WHERE ea.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_sql, selected_tasks)
            rows = self.db.cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            # データフレームに変換
            df = pd.DataFrame(rows, columns=['model', 'talk_nums', 'word_info_id', 'get_talk_nums'])

            # 不一致の判定
            df['Mismatch'] = df.apply(lambda row: row['talk_nums'] != row['get_talk_nums'], axis=1)

            # 不一致のあるデータのみを抽出
            mismatch_df = df[df['Mismatch']]

            # 各モデル・word_info_idごとの不一致のtalk_numsを収集
            result_df = mismatch_df.groupby(['model', 'word_info_id']).agg({
                'talk_nums': lambda x: ', '.join(map(str, x))
            }).reset_index()

            # 各モデルごとの合計数を計算
            model_totals = result_df.groupby('model').agg({
                'talk_nums': lambda x: sum(len(talk_nums.split(', ')) for talk_nums in x)
            }).reset_index().rename(columns={'talk_nums': 'Model_Total'})

            # 各 word_info_id ごとの合計数を計算
            word_totals = result_df.groupby('word_info_id').agg({
                'talk_nums': lambda x: sum(len(talk_nums.split(', ')) for talk_nums in x)
            }).reset_index().rename(columns={'talk_nums': 'Word_Total'})

            # 全体の合計数を計算
            grand_total = model_totals['Model_Total'].sum()

            # ピボットテーブルの作成
            pivot_df = result_df.pivot(index='model', columns='word_info_id', values='talk_nums')

            # 合計列を追加
            pivot_df['合計'] = model_totals.set_index('model')['Model_Total']

            # 合計行を追加
            pivot_df.loc['合計'] = word_totals.set_index('word_info_id')['Word_Total']
            pivot_df.at['合計', '合計'] = grand_total

            # NaN を空文字に置き換え
            pivot_df = pivot_df.fillna('')

            # インデックスをリセット
            pivot_df = pivot_df.reset_index()

            return pivot_df
        except Exception as e:
            print(f"Error fetching cross tab data: {e}")
            return pd.DataFrame()
        

    def perform_pca(self, selected_tasks):
        try:
            # 選択されたタスクに対応するベクトルと関連情報を取得
            vectors_list = []
            task_names = []
            words = []

            for task_name in selected_tasks:
                # ベクトルと関連情報を取得
                vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)
                if vectors.size == 0:
                    print(f"タスク名 '{task_name}' に対応するベクトルが見つかりませんでした。")
                    continue

                # ベクトルをリストに追加
                vectors_list.append(vectors)

                # タスク名とワードを取得
                # words_info テーブルから word を取得
                word_list = []
                for talk_num in talk_nums:
                    select_sql = '''
                    SELECT wi.word
                    FROM conversations c
                    INNER JOIN words_info wi ON c.word_info_id = wi.word_info_id
                    WHERE c.task_id = (SELECT task_id FROM tasks WHERE task_name = ?) AND c.talk_num = ?
                    '''
                    self.db.cursor.execute(select_sql, (task_name, talk_num))
                    row = self.db.cursor.fetchone()
                    word = row[0] if row else ''
                    word_list.append(word)

                task_names.extend([task_name] * len(vectors))
                words.extend(word_list)

            if not vectors_list:
                return pd.DataFrame(), None, None

            # 全てのベクトルを結合
            all_vectors = np.vstack(vectors_list)

            # 主成分分析の実行
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(all_vectors)

            # 結果をデータフレームに格納
            pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
            pca_df['task_name'] = task_names
            pca_df['word'] = words

            return pca_df, pca, None
        except Exception as e:
            print(f"Error performing PCA: {e}")
            return pd.DataFrame(), None, None

    def close(self):
        self.db.close()
        self.vector_db.close()