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
        """
        クロス集計を取得し、DataFrame 形式で返す。
        この中で rag_results から processing_time も取得して、平均処理時間を算出するようにする。
        """
        try:
            placeholders = ','.join(['?'] * len(selected_tasks))

            # 1) evaluated_answers
            select_ea_sql = f'''
            SELECT
                ea.model,
                ea.qa_id,
                ea.talk_nums,
                ea.task_name
            FROM evaluated_answers ea
            WHERE ea.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_ea_sql, selected_tasks)
            ea_rows = self.db.cursor.fetchall()
            if not ea_rows:
                print("No data fetched for evaluated_answers.")
                return pd.DataFrame()

            evaluated_answers = pd.DataFrame(ea_rows, columns=['model', 'qa_id', 'talk_nums', 'task_name'])

            # 2) rag_results
            select_rr_sql = f'''
            SELECT
                rr.rag_id,
                rr.model,
                rr.qa_id,
                rr.talk_num_1,
                rr.talk_num_2,
                rr.talk_num_3,
                rr.talk_num_4,
                rr.talk_num_5,
                rr.processing_time,  -- ★ processing_time を取得
                rr.task_name
            FROM rag_results rr
            WHERE rr.task_name IN ({placeholders})
            '''
            self.db.cursor.execute(select_rr_sql, selected_tasks)
            rr_rows = self.db.cursor.fetchall()
            if not rr_rows:
                print("No data fetched for rag_results.")
                return pd.DataFrame()

            rag_results = pd.DataFrame(rr_rows, columns=[
                'rag_id', 'model', 'qa_id',
                'talk_num_1', 'talk_num_2', 'talk_num_3', 'talk_num_4', 'talk_num_5',
                'processing_time',
                'task_name'
            ])

            # 3) エラー計算 + 平均処理時間
            result_df = self.calculate_errors(evaluated_answers, rag_results)
            return result_df

        except Exception as e:
            print(f"Error fetching cross tab data: {e}")
            return pd.DataFrame()
        

    def calculate_errors(self, evaluated_answers, rag_results):
        """
        各 (model, qa_id) ごとに talk_nums が一致しない場合をエラーとみなし、
        さらに rag_results の processing_time の平均を算出して出力する。
        """
        try:
            models = set(evaluated_answers['model']).union(set(rag_results['model']))
            results_rows = []

            # rag_results から (model, qa_id) 単位の情報をまとめる
            rag_dict = {}
            for _, rr_row in rag_results.iterrows():
                model = rr_row['model']
                qa_id = rr_row['qa_id']
                dict_key = (model, qa_id)

                if dict_key not in rag_dict:
                    rag_dict[dict_key] = {
                        'rag_id': rr_row['rag_id'],
                        'talk_nums': [],
                        'task_name': rr_row['task_name'],
                        'processing_times': []
                    }

                # 5つの talk_num
                tlist = [
                    str(rr_row['talk_num_1']).strip(),
                    str(rr_row['talk_num_2']).strip(),
                    str(rr_row['talk_num_3']).strip(),
                    str(rr_row['talk_num_4']).strip(),
                    str(rr_row['talk_num_5']).strip()
                ]
                tlist = [x for x in tlist if x and x != 'None']
                rag_dict[dict_key]['talk_nums'].extend(tlist)

                # processing_time
                pt = rr_row.get('processing_time', 0.0) or 0.0
                rag_dict[dict_key]['processing_times'].append(pt)

            # evaluated_answers を走査してエラー判定
            for _, ea_row in evaluated_answers.iterrows():
                model_val = ea_row['model']
                qa_id_val = ea_row['qa_id']
                task_name = ea_row['task_name']
                ea_talk   = str(ea_row['talk_nums']).strip()

                dict_key = (model_val, qa_id_val)
                if dict_key not in rag_dict:
                    # RAG結果がない → エラー
                    error_count = 1
                    rag_id_str = "N/A"
                    proc_times = []
                else:
                    info = rag_dict[dict_key]
                    rag_talks = info['talk_nums']
                    proc_times = info['processing_times']

                    if ea_talk not in rag_talks:
                        error_count = 1
                        rag_id_str = str(info['rag_id'])
                    else:
                        error_count = 0
                        rag_id_str = ""

                # 平均処理時間を計算
                if proc_times:
                    avg_time = sum(proc_times) / len(proc_times)
                else:
                    avg_time = 0.0

                results_rows.append({
                    'タスク名': task_name,
                    'モデル名': model_val,
                    'qa_id': qa_id_val,
                    'エラー箇所(rag_id)': rag_id_str,
                    'エラー数': error_count,
                    '平均処理時間': avg_time
                })

            # 個票レベルを df 化
            df = pd.DataFrame(results_rows)
            if df.empty:
                return df

            # 最終的に (タスク名, モデル名) ごとに集計
            agg_df = (
                df.groupby(['タスク名', 'モデル名'], as_index=False)
                .agg({
                    'qa_id': 'count',           # → 合計数
                    'エラー数': 'sum',          # → エラー合計数
                    'エラー箇所(rag_id)': lambda x: ','.join([str(v) for v in x if v]),
                    '平均処理時間': 'mean'      # → 平均処理時間の平均
                })
            )

            agg_df.rename(columns={
                'qa_id': '合計数',
                'エラー数': 'エラー合計数',
                'エラー箇所(rag_id)': 'エラー箇所(rag_id)',
                '平均処理時間': '処理時間(平均)'
            }, inplace=True)

            agg_df['エラー率'] = agg_df['エラー合計数'] / agg_df['合計数']
            return agg_df

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

                svd_results.append({
                    'qa_id': qa_id,
                    'question_text': question_text,
                    'expected_answer': expected_answer,
                    'model_answer': model_answer,
                    'coordinates': coordinates
                })

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




    def get_evaluation_results(self, task_name: str):
        """
        evaluated_answers と rag_results を LEFT JOIN し、
        - rag_results の rag_id も取得
        - get_talk_nums (カンマ区切り) に基づき、get_context_1..5 を生成
        """
        try:
            sql = """
            SELECT
                ea.qa_id,
                ea.task_name,
                ea.model,
                ea.question,
                ea.expected_answer,
                ea.gpt_response,
                ea.talk_nums,
                ea.get_talk_nums,

                gqa.word_info_id,
                wi.word,

                rr.processing_time AS rag_processing_time,
                rr.rag_id

            FROM evaluated_answers ea
            LEFT JOIN generated_qas gqa
                ON ea.qa_id = gqa.qa_id
            LEFT JOIN words_info wi
                ON gqa.word_info_id = wi.word_info_id
            LEFT JOIN rag_results rr
                ON ea.qa_id = rr.qa_id
                AND ea.task_id = rr.task_id
                AND ea.model = rr.model

            WHERE ea.task_name = ?
            ORDER BY ea.qa_id ASC
            """
            self.db.cursor.execute(sql, (task_name,))
            rows = self.db.cursor.fetchall()
            if not rows:
                return []

            col_names = [desc[0] for desc in self.db.cursor.description]
            results = []

            for row in rows:
                row_dict = dict(zip(col_names, row))

                # カンマ区切りを分割して talk_num のリスト化
                talk_num_str = row_dict.get("get_talk_nums", "")
                if talk_num_str:
                    talk_num_list = [x.strip() for x in talk_num_str.split(',') if x.strip()]
                else:
                    talk_num_list = []

                # 最大5件まで get_context_1..5 を埋め込む
                for i in range(5):
                    ctx_key = f"get_context_{i+1}"
                    if i < len(talk_num_list):
                        t_num = talk_num_list[i]

                        # conversationsテーブルを参照して発話を取得
                        context_lines = []
                        conv_sql = """
                        SELECT c.user, c.assistant
                        FROM conversations c
                        INNER JOIN tasks t ON c.task_id = t.task_id
                        WHERE t.task_name = ?
                        AND c.talk_num = ?
                        ORDER BY c.created_at ASC
                        """
                        self.db.cursor.execute(conv_sql, (task_name, t_num))
                        convo_rows = self.db.cursor.fetchall()

                        for (u, a) in convo_rows:
                            context_lines.append(f"User: {u}\nAssistant: {a}")

                        if context_lines:
                            merged_context = "\n\n".join(context_lines)
                        else:
                            merged_context = "(該当会話なし)"

                        row_dict[ctx_key] = merged_context
                    else:
                        # 5件に満たない場合は空文字
                        row_dict[ctx_key] = ""

                results.append(row_dict)

            return results

        except Exception as e:
            print(f"Error in get_evaluation_results: {e}")
            return []


    def close(self):
        self.db.close()
        self.vector_db.close()
