# vector_db.py

import sqlite3
import openai
import numpy as np
import faiss
import os
from database import ConversationDatabase
from sklearn.decomposition import PCA
from typing import Any, Dict, Tuple, List
import pickle
import constants as c

class VectorDatabase:
    def __init__(self, api_key, db_file='conversation_data.db', index_folder='faiss_index', 
                 index_file='faiss_index.index', pca_model_file='pca_model.pkl', 
                 max_components=768, min_components=5):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file
        
        # PCAの設定を動的に
        self.max_components = max_components
        self.min_components = min_components
        
        # インデックスフォルダのパスを設定
        self.index_folder = index_folder
        self.index_file = index_file  # インデックスファイル名
        self.pca_model_file = os.path.join(self.index_folder, pca_model_file)  # PCAモデルのファイルパス

        # インデックスフォルダが存在しない場合は作成
        if not os.path.exists(self.index_folder):
            os.makedirs(self.index_folder)

        # インデックスファイルのフルパスを設定
        self.index_file_path = os.path.join(self.index_folder, self.index_file)

        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        self.create_vector_table()
        self.index = None  # FAISS インデックス
        self.ids = []      # ベクトルの ID リスト

        # PCAモデルの初期化
        self.pca = None
        self.load_pca_model()
        self.load_faiss_index()

    def create_vector_table(self):
        # vector_table テーブルの作成
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_table (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            word_info_id INTEGER NOT NULL,
            talk_num TEXT NOT NULL,
            content TEXT,
            row_vector BLOB,
            vector BLOB,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        )
        ''')
        self.conn.commit()

    def compute_query_embedding(self, query: str) -> np.ndarray:
        """
        クエリをベクトル化し、PCA を適用して正規化します。
        """
        # クエリをベクトル化
        response = openai.Embedding.create(
            input=query,
            model=c.EMBEDDING_MODEL
        )
        query_vector = np.array(response['data'][0]['embedding'], dtype='float32')

        # PCA の適用
        if self.pca is not None:
            query_vector = self.pca.transform(query_vector.reshape(1, -1))[0]
        else:
            print("PCA モデルがロードされていません。")

        # ベクトルの正規化
        norm = np.linalg.norm(query_vector)
        if norm != 0:
            query_vector = query_vector / norm
        else:
            print("クエリベクトルのノルムが 0 です。")

        return query_vector.astype('float32')

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        コサイン類似度を計算します。
        """
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # ゼロノルム対策
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def load_pca_model(self):
        # PCA モデルのロード
        if os.path.exists(self.pca_model_file):
            with open(self.pca_model_file, 'rb') as f:
                self.pca = pickle.load(f)
            print(f"PCA モデルを '{self.pca_model_file}' から読み込みました。")
        else:
            print(f"PCA モデルファイル '{self.pca_model_file}' が存在しません。")
            self.pca = None

    def load_faiss_index(self):
        if os.path.exists(self.index_file_path):
            self.index = faiss.read_index(self.index_file_path)
            print(f"FAISS インデックスを '{self.index_file_path}' から読み込みました。")
        else:
            print(f"FAISS インデックスファイル '{self.index_file_path}' が存在しません。")
            self.index = None

    def create_faiss_index(self, vectors, ids):
        # ベクトルの正規化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        dimension = vectors.shape[1]
        # 内積を使用したインデックスを作成
        index = faiss.IndexFlatIP(dimension)
        # ID を指定するために IndexIDMap を使用
        index = faiss.IndexIDMap(index)
        # インデックスにベクトルを追加
        index.add_with_ids(vectors.astype('float32'), np.array(ids))

        # インデックスをファイルに保存
        faiss.write_index(index, self.index_file_path)
        print(f"FAISS インデックスを '{self.index_file_path}' に保存しました。")

    def vectorize_conversations(self, task_name):
        # ConversationDatabase を使用して会話データを取得
        convo_db = ConversationDatabase(db_file=self.db_file)
        conversations = convo_db.get_conversations_with_task_name(task_name)
        convo_db.close()

        if not conversations:
            raise ValueError(f"タスク名 '{task_name}' に一致する会話が見つかりません。")

        row_vectors = []
        vector_ids = []
        data_to_insert = []

        # 会話データを task_id, word_info_id, talk_num で一意に絞り込む
        unique_conversations = {}
        for convo in conversations:
            key = (convo['task_id'], convo['word_info_id'], convo['talk_num'])
            if key not in unique_conversations:
                unique_conversations[key] = []
            unique_conversations[key].append(convo)

        for key, convos in unique_conversations.items():
            task_id, word_info_id, talk_num = key

            # ユーザーとアシスタントの発話を時系列順に並べる
            sorted_convos = sorted(convos, key=lambda x: int(x['talk_num']))

            # ユーザーとアシスタントの発話を結合
            content_parts = []
            for convo in sorted_convos:
                user = convo['user']
                assistant = convo['assistant']
                content_parts.append(f"User: {user}\nAssistant: {assistant}")

            content = "\n".join(content_parts)

            try:
                # 生のベクトルの取得
                response = openai.Embedding.create(
                    input=content,
                    model=c.EMBEDDING_MODEL
                )
                row_vector = response['data'][0]['embedding']
                row_vector_np = np.array(row_vector, dtype='float32')
                row_vector_bytes = row_vector_np.tobytes()

                # データを一時的に保存
                data_to_insert.append({
                    'task_id': task_id,
                    'word_info_id': word_info_id,
                    'talk_num': talk_num,
                    'content': content,
                    'row_vector': row_vector_np,
                    'row_vector_bytes': row_vector_bytes
                })

                # row_vector をリストに追加（PCA の適用に使用）
                row_vectors.append(row_vector_np)

            except Exception as e:
                print(f"ベクトル化中にエラーが発生しました (task_id: {task_id}, word_info_id: {word_info_id}, talk_num: {talk_num}): {e}")
                continue

        # PCA の適用
        if row_vectors:
            row_vectors_np = np.array(row_vectors)
            n_samples = len(row_vectors)
            
            # 動的にn_componentsを決定
            n_components = min(max(self.min_components, n_samples - 1), self.max_components)
            
            print(f"データ数: {n_samples}, 使用する次元数: {n_components}")

            # PCA モデルのロードまたはフィッティング
            if self.pca is None:
                self.pca = PCA(n_components=n_components)
                self.pca.fit(row_vectors_np)
                # PCA モデルを保存
                with open(self.pca_model_file, 'wb') as f:
                    pickle.dump(self.pca, f)
                print(f"PCA モデルを '{self.pca_model_file}' に保存しました。")

            # PCA を適用してベクトルを取得
            vector_np_all = self.pca.transform(row_vectors_np)

            # ベクトルの正規化
            norms = np.linalg.norm(vector_np_all, axis=1, keepdims=True)
            vector_np_all = vector_np_all / norms

            # データベースへのデータ挿入と FAISS インデックスへの追加
            vectors = []
            ids = []

            for idx, data in enumerate(data_to_insert):
                try:
                    # PCA 適用後のベクトルを取得
                    vector_np = vector_np_all[idx]
                    vector_bytes = vector_np.tobytes()

                    # データベースに挿入
                    insert_sql = '''
                    INSERT INTO vector_table (task_id, word_info_id, talk_num, content, row_vector, vector)
                    VALUES (?, ?, ?, ?, ?, ?)
                    '''
                    insert_data = (
                        data['task_id'],
                        data['word_info_id'],
                        data['talk_num'],
                        data['content'],
                        data['row_vector_bytes'],
                        vector_bytes
                    )
                    self.cursor.execute(insert_sql, insert_data)
                    self.conn.commit()

                    # vector_id を取得
                    vector_id = self.cursor.lastrowid

                    # FAISS インデックスに追加
                    vectors.append(vector_np)
                    ids.append(vector_id)

                except Exception as e:
                    print(f"データベースへの挿入中にエラーが発生しました (task_id: {data['task_id']}, word_info_id: {data['word_info_id']}, talk_num: {data['talk_num']}): {e}")
                    continue

            # FAISS インデックスを作成してファイルに保存
            if vectors:
                self.create_faiss_index(np.array(vectors), ids)

        return len(unique_conversations)

    def fetch_vectors_and_contents(self, task_name: str) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
        """
        特定のタスクに対応するベクトルとコンテンツをデータベースから取得します。
        """
        select_sql = '''
        SELECT v.vector_id, v.content, v.vector, v.task_id, v.word_info_id, v.talk_num
        FROM vector_table v
        INNER JOIN conversations c ON v.task_id = c.task_id AND v.word_info_id = c.word_info_id AND v.talk_num = c.talk_num
        INNER JOIN tasks t ON c.task_id = t.task_id
        WHERE t.task_name = ?
        '''
        self.cursor.execute(select_sql, (task_name,))
        rows = self.cursor.fetchall()
        if not rows:
            raise ValueError(f"タスク名 '{task_name}' に対応するベクトルが存在しません。")

        vectors = []
        contents = []
        vector_ids = []
        talk_nums = []
        for row in rows:
            vector_id, content, vector_blob, task_id, word_info_id, talk_num = row
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            vectors.append(vector)
            contents.append(content)
            vector_ids.append(vector_id)
            talk_nums.append(talk_num)

        return np.array(vectors), contents, vector_ids, talk_nums

    def retrieve_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        クエリベクトルに類似したベクトルを FAISS インデックスから検索します。
        常にtop_k件（または全件数の少ない方）の結果を返します。
        """
        if self.index is None:
            print("FAISS インデックスがロードされていません。")
            return []

        # インデックス内の総ベクトル数を取得
        total_vectors = self.index.ntotal
        print(f"インデックス内のベクトル数: {total_vectors}")

        # 実際に取得する件数を決定
        actual_k = min(top_k, total_vectors)
        
        # 検索の実行
        distances, indices = self.index.search(query_vector.reshape(1, -1), actual_k)

        # すべての結果をログ出力
        results = [(idx, dist) for dist, idx in zip(distances[0], indices[0])]
        print(f"検索結果 - 件数: {len(results)}")
        for idx, dist in results:
            print(f"vector_id: {idx}, 類似度スコア: {dist}")

        return results

    def close(self):
        self.conn.close()

def vectorize_and_store(api_key, task_name):
    vector_db = VectorDatabase(api_key=api_key)
    try:
        added_count = vector_db.vectorize_conversations(task_name=task_name)
        return added_count
    finally:
        vector_db.close()