# analysis.py

from database import ConversationDatabase
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import joblib  # PCAモデルの保存と読み込みに使用
from vector_db import VectorDatabase

class Analysis:
    def __init__(self, api_key, db_file='conversation_data.db', index_folder='faiss_index',
                 index_file='faiss_index.index', pca_model_file='pca_model.joblib'):
        self.db = ConversationDatabase(db_file=db_file)
        self.vector_db = VectorDatabase(api_key=api_key, db_file=db_file, index_folder=index_folder, index_file=index_file)
        self.pca_model_file = pca_model_file  # PCAモデルのファイルパス

    def save_pca_model(self, pca):
        joblib.dump(pca, self.pca_model_file)

    def load_pca_model(self):
        if os.path.exists(self.pca_model_file):
            return joblib.load(self.pca_model_file)
        else:
            return None

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
            placeholders = ','.join(['?'] * len(selected_tasks))

            # Fetch evaluated_answers data
            select_ea_sql = f'''
            SELECT
                ea.model,
                ea.qa_id,
                ea.talk_nums
            FROM evaluated_answers ea
            WHERE ea.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_ea_sql, selected_tasks)
            ea_rows = self.db.cursor.fetchall()

            if not ea_rows:
                print("No data fetched for evaluated_answers.")
                return pd.DataFrame()

            # Convert evaluated_answers to DataFrame
            evaluated_answers = pd.DataFrame(ea_rows, columns=['model', 'qa_id', 'talk_nums'])

            # Fetch rag_results data
            select_rr_sql = f'''
            SELECT
                rr.model,
                rr.qa_id,
                rr.talk_num_1,
                rr.talk_num_2,
                rr.talk_num_3,
                rr.talk_num_4,
                rr.talk_num_5
            FROM rag_results rr
            WHERE rr.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_rr_sql, selected_tasks)
            rr_rows = self.db.cursor.fetchall()

            if not rr_rows:
                print("No data fetched for rag_results.")
                return pd.DataFrame()

            # Convert rag_results to DataFrame
            rag_results = pd.DataFrame(rr_rows, columns=[
                'model', 'qa_id', 'talk_num_1', 'talk_num_2', 'talk_num_3', 'talk_num_4', 'talk_num_5'
            ])

            # Call calculate_errors_simple
            result_df = self.calculate_errors(evaluated_answers, rag_results)

            return result_df

        except Exception as e:
            print(f"Error fetching cross tab data: {e}")
            return pd.DataFrame()


    def calculate_errors(self, evaluated_answers, rag_results):
        try:
            # モデル名のリストを取得
            models = set(evaluated_answers['model']).union(set(rag_results['model']))

            # 結果を格納するリストを初期化
            results = []

            for model in models:
                # 特定のモデルのデータをフィルタリング
                eval_answers_model = evaluated_answers[evaluated_answers['model'] == model]
                rag_results_model = rag_results[rag_results['model'] == model]

                # evaluated_answers を辞書に変換: {qa_id: talk_num}
                eval_dict = {
                    row['qa_id']: str(row['talk_nums']).strip()
                    for _, row in eval_answers_model.iterrows()
                }

                # rag_results を辞書に変換: {qa_id: [talk_num_1, ..., talk_num_5]}
                rag_dict = {}
                for _, row in rag_results_model.iterrows():
                    qa_id = row['qa_id']
                    rag_talk_nums = [
                        str(row['talk_num_1']).strip(),
                        str(row['talk_num_2']).strip(),
                        str(row['talk_num_3']).strip(),
                        str(row['talk_num_4']).strip(),
                        str(row['talk_num_5']).strip()
                    ]
                    # None や空文字列を除外
                    rag_talk_nums = [t for t in rag_talk_nums if t and t != 'None']
                    rag_dict[qa_id] = rag_talk_nums

                # エラーの追跡
                errors = []
                total_count = len(eval_dict)
                for qa_id, talk_num in eval_dict.items():
                    rag_talk_nums = rag_dict.get(qa_id, [])
                    if talk_num not in rag_talk_nums:
                        errors.append(qa_id)

                # 結果の計算
                error_count = len(errors)
                error_rate = error_count / total_count if total_count > 0 else 0

                # 結果をリストに追加
                results.append({
                    'モデル名': model,
                    'エラー箇所': ','.join(map(str, errors)),
                    'エラー合計数': error_count,
                    '合計数': total_count,
                    'エラー率': error_rate
                })

            # 結果のデータフレームを作成
            result_df = pd.DataFrame(results)
            return result_df

        except Exception as e:
            print(f"Error in calculate_errors: {e}")
            return pd.DataFrame()



    def perform_pca(self, selected_tasks):
        try:
            # 選択されたタスクに対応するベクトルと関連情報を取得
            vectors_list = []
            task_names = []
            words = []
            vector_ids_list = []
            pca_vectors_list = []

            for task_name in selected_tasks:
                try:
                    # ベクトルと関連情報を取得
                    vectors, contents, vector_ids, talk_nums, pca_vectors = self.vector_db.fetch_vectors_and_pca_vectors(task_name)
                except ValueError as e:
                    print(str(e))
                    continue  # 次のタスクへ

                if vectors.size == 0:
                    print(f"タスク名 '{task_name}' に対応するベクトルが見つかりませんでした。")
                    continue

                vectors_list.append(vectors)
                vector_ids_list.extend(vector_ids)
                pca_vectors_list.extend(pca_vectors)

                # タスク名とワードを取得
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
                print("選択されたタスクに対応するデータが見つかりませんでした。")
                return pd.DataFrame(), None, None

            # 全てのベクトルを結合
            all_vectors = np.vstack(vectors_list)

            # PCAモデルをロードまたは新たに作成
            pca = self.load_pca_model()
            if pca is None:
                print("PCAモデルが存在しません。新たにPCAを計算します。")
                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(all_vectors)
                # PCAモデルを保存
                self.save_pca_model(pca)
                # PCAベクトルをデータベースに保存
                for vector_id, pca_vector in zip(vector_ids_list, principalComponents):
                    pca_vector_bytes = pca_vector.tobytes()
                    update_sql = 'UPDATE vector_table SET pca_vector = ? WHERE vector_id = ?'
                    self.db.cursor.execute(update_sql, (pca_vector_bytes, vector_id))
                self.db.conn.commit()
            else:
                # 既存のPCAモデルを使用
                principalComponents = pca.transform(all_vectors)
                # pca_vector が存在しないデータに対してのみ保存
                for vector_id, pca_vector, existing_pca_vector in zip(vector_ids_list, principalComponents, pca_vectors_list):
                    if existing_pca_vector is None:
                        pca_vector_bytes = pca_vector.tobytes()
                        update_sql = 'UPDATE vector_table SET pca_vector = ? WHERE vector_id = ?'
                        self.db.cursor.execute(update_sql, (pca_vector_bytes, vector_id))
                self.db.conn.commit()

            # 結果をデータフレームに格納
            pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
            pca_df['task_name'] = task_names
            pca_df['word'] = words

            return pca_df, pca, None
        except Exception as e:
            print(f"Error performing PCA: {e}")
            return pd.DataFrame(), None, None
        


    def calculate_drift_direction(self, selected_tasks):
        try:
            results = []
            for task_name in selected_tasks:
                # ベクトルを取得
                select_sql = '''
                SELECT v.query_vector, v.gold_answer_vector, v.rag_answer_vector, wi.word
                FROM vector_table v
                INNER JOIN tasks t ON v.task_id = t.task_id
                INNER JOIN words_info wi ON v.word_info_id = wi.word_info_id
                WHERE t.task_name = ? AND v.query_vector IS NOT NULL AND v.gold_answer_vector IS NOT NULL AND v.rag_answer_vector IS NOT NULL
                '''
                self.db.cursor.execute(select_sql, (task_name,))
                rows = self.db.cursor.fetchall()
                if not rows:
                    continue  # データがない場合は次のタスクへ

                for row in rows:
                    query_vector_blob, gold_vector_blob, rag_vector_blob, word = row
                    if not query_vector_blob or not gold_vector_blob or not rag_vector_blob:
                        continue  # ベクトルが欠損している場合はスキップ

                    query_vector = np.frombuffer(query_vector_blob, dtype='float32')
                    gold_vector = np.frombuffer(gold_vector_blob, dtype='float32')
                    rag_vector = np.frombuffer(rag_vector_blob, dtype='float32')

                    # ベクトル差分を計算
                    V1 = gold_vector - query_vector
                    V2 = rag_vector - gold_vector

                    # コサイン類似度を計算
                    cosine_sim = self.cosine_similarity(V1, V2)

                    results.append({
                        'task_name': task_name,
                        'word': word,
                        'cosine_similarity': cosine_sim
                    })
            return pd.DataFrame(results)
        except Exception as e:
            print(f"Error in calculate_drift_direction: {e}")
            return pd.DataFrame()


    def close(self):
        self.db.close()
        self.vector_db.close()
