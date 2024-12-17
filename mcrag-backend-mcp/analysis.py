# analysis.py

from database import ConversationDatabase
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import joblib  # PCAモデルの保存と読み込みに使用
from vector_db import VectorDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import constants as c  # constants.py をインポート
import openai



class Analysis:
    def __init__(self, api_key, db_file='conversation_data.db', index_folder='faiss_index',
                 index_file='faiss_index.index', pca_model_file='pca_model.joblib'):        
        self.db = ConversationDatabase(db_file=db_file)
        self.vector_db = VectorDatabase(api_key=api_key, db_file=db_file, index_folder=index_folder, index_file=index_file)
        self.api_key = api_key
        openai.api_key = self.api_key
        self.pca_model_file = pca_model_file  # PCAモデルのファイルパス
        self.pca = self.load_pca_model()
        if self.pca is None:
            self.pca = PCA(n_components=2)

    def save_pca_model(self):
        joblib.dump(self.pca, self.pca_model_file)

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
            placeholders = ','.join(['?'] * len(selected_tasks))

            select_sql = f'''
            SELECT ea.qa_id, ea.task_id, ea.task_name, gq.question, ea.gpt_response, ea.expected_answer, wi.word, ea.model
            FROM evaluated_answers ea
            INNER JOIN generated_qas gq ON ea.qa_id = gq.qa_id
            INNER JOIN words_info wi ON gq.word_info_id = wi.word_info_id
            WHERE ea.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_sql, selected_tasks)
            rows = self.db.cursor.fetchall()

            if not rows:
                print("選択されたタスクに対応するデータが見つかりませんでした。")
                return []

            vectors = []
            labels = []
            qa_info = []
            processed_expected = set()

            # 一時的に期待値・モデル回答データを格納するための構造
            data_map = {} # {qa_id: {"expected": (expected_vector, info_dict), "models": [(model_vector, model, info_dict), ...]}}

            for row in rows:
                qa_id, task_id, task_name, question_text, model_answer, expected_answer, word, model = row

                if not model_answer or not expected_answer:
                    continue

                expected_vector, actual_vector = self.db.get_pca_vectors_by_qa_id_and_model(qa_id, model)
                if expected_vector is None or actual_vector is None:
                    # ベクトル化
                    actual_vector = self.get_embedding(model_answer)
                    if qa_id not in processed_expected:
                        expected_vector = self.get_embedding(expected_answer)
                    else:
                        # 既に期待値はDBに保存済みと想定可能
                        expected_vector = self.get_embedding(expected_answer)

                    # DB保存
                    self.db.insert_pca_analysis(
                        task_id=task_id,
                        task_name=task_name,
                        qa_id=qa_id,
                        model=model,
                        expected_vector=expected_vector,
                        actual_vector=actual_vector,
                        distance=None  # ここでは一旦Noneを入れて後で更新してもいいし、計算後再INSERTしてもOK
                    )

                # data_mapに格納
                if qa_id not in data_map:
                    data_map[qa_id] = {
                        "task_id": task_id,
                        "task_name": task_name,
                        "question_text": question_text,
                        "expected_answer": expected_answer,
                        "word": word,
                        "expected_vector": expected_vector,
                        "models": []
                    }

                data_map[qa_id]["models"].append((actual_vector, model_answer, model))
                processed_expected.add(qa_id)


            # デバッグ用print：data_mapのサイズ
            print("data_map size:", len(data_map))
            # PCA用のデータを整形
            # 期待値 -> 1行, モデル -> n行
            for qa_id, val in data_map.items():
                print("QA ID:", qa_id, "Models count:", len(val["models"]))

                exp_vec = val["expected_vector"]
                print("Appending QA info (expected):", qa_id, "model:", "expected")

                vectors.append(exp_vec)
                labels.append('期待値')
                qa_info.append({
                    'qa_id': qa_id,
                    'task_id': val["task_id"],  # ここでtask_idを追加
                    'task_name': val["task_name"],
                    'answer_type': '期待値',
                    'question_text': val["question_text"],
                    'expected_answer': val["expected_answer"],
                    'model_answer': val["expected_answer"],
                    'word': val["word"],
                    'model': 'expected'
                })

                for (act_vec, m_answer, m_model) in val["models"]:
                    print("Appending QA info (model):", qa_id, "model:", m_model)
                    vectors.append(act_vec)
                    labels.append(f'モデル: {m_model}')
                    qa_info.append({
                        'qa_id': qa_id,
                        'task_id': val["task_id"],  # ここでもtask_idを追加
                        'task_name': val["task_name"],
                        'answer_type': f'モデル: {m_model}',
                        'question_text': val["question_text"],
                        'expected_answer': val["expected_answer"],
                        'model_answer': m_answer,
                        'word': val["word"],
                        'model': m_model
                    })
            # デバッグ用print
            print("Length of vectors:", len(vectors))
            print("Length of labels:", len(labels))
            print("Length of qa_info:", len(qa_info))

            if not vectors:
                print("有効なベクトルデータがありません。")
                return []

            vectors = np.array(vectors)

            principalComponents = self.pca.fit_transform(vectors)
            self.save_pca_model()
            print("principalComponents shape:", principalComponents.shape)


            pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
            qa_info_df = pd.DataFrame(qa_info)
            print("pca_df shape before concat:", pca_df.shape)
            print("qa_info_df shape:", qa_info_df.shape)

            pca_df = pd.concat([pca_df.reset_index(drop=True), qa_info_df.reset_index(drop=True)], axis=1)

            print("pca_df shape after concat:", pca_df.shape)           
            pca_df['Label'] = labels

            # distance計算
            pca_df['distance'] = None

            # qa_idごとに期待値行とモデル行との距離を計算
            for qa_id_val, gdf in pca_df.groupby('qa_id'):
                expected_row = gdf[gdf['answer_type'] == '期待値']
                model_rows = gdf[gdf['answer_type'].str.startswith('モデル:')]

                if len(expected_row) == 1:
                    exp_pc1 = expected_row['PC1'].values[0]
                    exp_pc2 = expected_row['PC2'].values[0]

                    for idx, mrow in model_rows.iterrows():
                        model_pc1 = mrow['PC1']
                        model_pc2 = mrow['PC2']
                        dist = ((exp_pc1 - model_pc1)**2 + (exp_pc2 - model_pc2)**2)**0.5
                        pca_df.at[idx, 'distance'] = dist

                        # DB更新
                        qa_id_db = mrow['qa_id']
                        model_db = mrow['model']

                        # すでにinsertしているデータがあるため、UPDATEかINSERT OR REPLACEでdistanceを更新可能
                        # ここでは再度insert_pca_analysisで上書きする想定(qa_id,model一意でREPLACE)
                        # 実装ではDISTANCEの更新専用UPDATEメソッドを作成してもよい
                        self.db.insert_pca_analysis(
                            task_id=mrow['task_id'],
                            task_name=mrow['task_name'],
                            qa_id=qa_id_db,
                            model=model_db,
                            expected_vector=None,  # Noneで既存値維持にはupdate文が必要
                            actual_vector=None,
                            distance=dist
                        )

            result = pca_df.to_dict(orient='records')
            return result

        except Exception as e:
            print(f"PCA分析中にエラーが発生しました: {e}")
            return []



    def perform_svd_analysis(self, selected_tasks):
        try:
            placeholders = ','.join(['?'] * len(selected_tasks))
            print("a", flush=True)

            # evaluated_answers テーブルからデータを取得
            select_sql = f'''
            SELECT ea.qa_id, ea.task_id, ea.task_name, ea.question, ea.gpt_response, ea.expected_answer
            FROM evaluated_answers ea
            WHERE ea.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_sql, selected_tasks)
            rows = self.db.cursor.fetchall()

            if not rows:
                print("選択されたタスクに対応するデータが見つかりませんでした。")
                return []
            print("b", flush=True)

            svd_results = []

            for row in rows:
                qa_id, task_id, task_name, question_text, model_answer, expected_answer = row

                if not model_answer or not expected_answer:
                    continue

                # 既にベクトルが保存されているかチェック
                expected_vector, actual_vector = self.db.get_svd_vectors_by_qa_id(qa_id)

                if expected_vector is None or actual_vector is None:
                    # ベクトルが存在しない場合、新しくベクトル化して保存
                    # モデルの回答をEmbeddingベクトル化
                    actual_vector = self.get_embedding(model_answer)

                    # 期待値をEmbeddingベクトル化
                    expected_vector = self.get_embedding(expected_answer)

                    # データベースに保存
                    self.db.insert_svd_analysis(
                        task_id=task_id,
                        task_name=task_name,
                        qa_id=qa_id,
                        expected_vector=expected_vector,
                        actual_vector=actual_vector
                    )

                # 2つのベクトルをスタック
                X = np.vstack([expected_vector, actual_vector])

                # SVDの実行
                svd = TruncatedSVD(n_components=2)
                svd_result = svd.fit_transform(X)
                print("c", flush=True)


                # プロット用のデータを準備
                coordinates = []
                labels = ['expected', 'actual']
                for i in range(2):
                    coord = {
                        'expected': float(svd_result[i, 0]),
                        'actual': float(svd_result[i, 1]),
                        'Label': labels[i]
                    }
                    coordinates.append(coord)
                print("d", flush=True)

                svd_results.append({
                    'qa_id': qa_id,
                    'question_text': question_text,
                    'expected_answer': expected_answer,
                    'model_answer': model_answer,
                    'coordinates': coordinates
                })
                print("e", flush=True)

            return svd_results

        except Exception as e:
            print(f"SVD分析中にエラーが発生しました: {e}")
            return []


    def get_embedding(self, text):
        """
        テキストをOpenAIのEmbeddingモデルを使用してベクトル化します。
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model=c.EMBEDDING_MODEL
            )
            embedding = response['data'][0]['embedding']
            return np.array(embedding, dtype='float32')
        except Exception as e:
            print(f"Embedding取得中にエラーが発生しました: {e}")
            return np.zeros((1536,), dtype='float32')  # モデルの次元数に合わせてゼロベクトルを返す




    def close(self):
        self.db.close()
        self.vector_db.close()
