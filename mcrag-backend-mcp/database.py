# database.py

import sqlite3
import logging
from datetime import datetime
import numpy as np

# ログの基本設定
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

# 例外クラスの定義
class DataAlreadyExistsError(Exception):
    """データが既に存在する場合の例外クラス"""
    pass

class DataNotFoundError(Exception):
    """データが見つからない場合の例外クラス"""
    pass

class ConversationDatabase:
    def __init__(self, db_file='conversation_data.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.conn.execute("PRAGMA foreign_keys = ON;")  # 外部キー制約を有効化
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # tasks テーブル
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL UNIQUE,
            character_prompt TEXT,
            user_prompt TEXT
        )
        ''')
        # words_info テーブル
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS words_info (
            word_info_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            word TEXT,
            info TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        )
        ''')
        # conversations テーブル
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            word_info_id INTEGER NOT NULL,
            talk_num TEXT NOT NULL,
            user TEXT,
            assistant TEXT,
            before_user TEXT,
            before_assistant TEXT,
            input_token INTEGER,
            output_token INTEGER,
            processing_time TEXT,
            temperature TEXT,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (word_info_id) REFERENCES words_info (word_info_id),
            UNIQUE (task_id, word_info_id, talk_num)
        )
        ''')
        # generated_qas テーブルの作成
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_qas (
            qa_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL,
            task_id INTEGER NOT NULL,
            word_info_id INTEGER NOT NULL,
            talk_nums TEXT NOT NULL,
            question TEXT,
            answer TEXT,
            input_token INTEGER,
            output_token INTEGER,
            processing_time REAL,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (word_info_id) REFERENCES words_info (word_info_id),
            UNIQUE (task_name, word_info_id, talk_nums)  -- 修正後の UNIQUE 制約
        )
        ''')
        # evaluated_answers テーブルの作成
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluated_answers (
            eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            qa_id INTEGER NOT NULL,
            task_name TEXT NOT NULL,
            task_id INTEGER NOT NULL,
            word_info_id INTEGER NOT NULL,
            talk_nums TEXT NOT NULL,
            question TEXT,
            expected_answer TEXT,
            gpt_response TEXT,
            get_context TEXT,
            get_talk_nums TEXT,
            input_token INTEGER,
            output_token INTEGER,            
            processing_time REAL,
            model TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (word_info_id) REFERENCES words_info (word_info_id),
            UNIQUE (task_name, word_info_id, talk_nums, model)
        )
        ''')
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
            pca_vector BLOB,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id)
        )
        ''')
        # rag_results テーブルの作成
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_results (
            rag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            qa_id INTEGER NOT NULL,
            task_name TEXT NOT NULL,
            task_id INTEGER NOT NULL,
            talk_num_1 TEXT NOT NULL,
            talk_num_2 TEXT NOT NULL,
            talk_num_3 TEXT NOT NULL,
            talk_num_4 TEXT NOT NULL,
            talk_num_5 TEXT NOT NULL,
            cosine_similarity_1 REAL,
            cosine_similarity_2 REAL,
            cosine_similarity_3 REAL,
            cosine_similarity_4 REAL,
            cosine_similarity_5 REAL,
            BM25_score_1 REAL,
            BM25_score_2 REAL,
            BM25_score_3 REAL,
            BM25_score_4 REAL,
            BM25_score_5 REAL,
            rss_rank_1 REAL,
            rss_rank_2 REAL,
            rss_rank_3 REAL,
            rss_rank_4 REAL,
            rss_rank_5 REAL,
            rerank_score_1 REAL,
            rerank_score_2 REAL,
            rerank_score_3 REAL,
            rerank_score_4 REAL,
            rerank_score_5 REAL,
            processing_time REAL NOT NULL,
            model TEXT NOT NULL,
            input_token_count INTEGER,
            output_token_count INTEGER,
            subject TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (qa_id) REFERENCES generated_qas (qa_id)
        )
        ''')
        self.conn.commit()
        # svd_analysis テーブルの作成
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS svd_analysis (
            svd_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            task_name TEXT NOT NULL,
            qa_id INTEGER NOT NULL,
            expected_vector BLOB,
            actual_vector BLOB,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (qa_id) REFERENCES generated_qas (qa_id)
        )
        ''')
        self.conn.commit()
        # pca_analysis テーブル (distance カラムを最初から含める)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS pca_analysis (
            pca_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            task_name TEXT NOT NULL,
            qa_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            expected_vector BLOB,
            actual_vector BLOB,
            distance REAL,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (qa_id) REFERENCES generated_qas (qa_id)
        )
        ''')
        self.conn.commit()

    # pca_analysis テーブルへのデータ挿入メソッドを修正
    def insert_pca_analysis(self, task_id, task_name, qa_id, model, expected_vector, actual_vector, distance=None):
        insert_sql = '''
        INSERT OR REPLACE INTO pca_analysis (
            task_id, task_name, qa_id, model, expected_vector, actual_vector, distance, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            task_id,
            task_name,
            qa_id,
            model,
            expected_vector.tobytes() if expected_vector is not None else None,
            actual_vector.tobytes() if actual_vector is not None else None,
            distance,
            self.get_current_timestamp()
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except Exception as e:
            print(f"pca_analysis テーブルへの挿入中にエラーが発生しました: {e}")

    def get_pca_vectors_by_qa_id_and_model(self, qa_id, model):
        select_sql = '''
        SELECT expected_vector, actual_vector FROM pca_analysis WHERE qa_id = ? AND model = ?
        '''
        try:
            self.cursor.execute(select_sql, (qa_id, model))
            row = self.cursor.fetchone()
            if row:
                expected_vector = np.frombuffer(row[0], dtype=np.float32) if row[0] else None
                actual_vector = np.frombuffer(row[1], dtype=np.float32) if row[1] else None
                return expected_vector, actual_vector
            else:
                return None, None
        except Exception as e:
            print(f"pca_analysis テーブルからの取得中にエラーが発生しました: {e}")
            return None, None

    def get_svd_vectors_by_qa_id(self, qa_id):
        """
        指定された qa_id の expected_vector と actual_vector を取得します。
        """
        select_sql = '''
        SELECT expected_vector, actual_vector FROM svd_analysis WHERE qa_id = ?
        '''
        try:
            self.cursor.execute(select_sql, (qa_id,))
            row = self.cursor.fetchone()
            if row:
                expected_vector = np.frombuffer(row[0], dtype=np.float32) if row[0] else None
                actual_vector = np.frombuffer(row[1], dtype=np.float32) if row[1] else None
                return expected_vector, actual_vector
            else:
                return None, None
        except Exception as e:
            logging.error(f"svd_analysis テーブルからの取得中にエラーが発生しました: {e}")
            return None, None
        
    # svd_analysis テーブルへのデータ挿入メソッドを追加
    def insert_svd_analysis(self, task_id, task_name, qa_id, expected_vector, actual_vector):
        insert_sql = '''
        INSERT INTO svd_analysis (
            task_id, task_name, qa_id, expected_vector, actual_vector, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        '''
        data = (
            task_id,
            task_name,
            qa_id,
            expected_vector.tobytes() if expected_vector is not None else None,
            actual_vector.tobytes() if actual_vector is not None else None,
            self.get_current_timestamp()
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except Exception as e:
            logging.error(f"svd_analysis テーブルへの挿入中にエラーが発生しました: {e}")

    def get_tasks_character_prompt(self, task_name):
        select_sql = 'SELECT character_prompt FROM tasks WHERE task_name = ?'
        self.cursor.execute(select_sql, (task_name,))
        result = self.cursor.fetchone()
        if result and result[0]:
            return result[0]
        else:
            raise ValueError(f"タスク '{task_name}' に対応する character_prompt が見つかりません。")
        
    def get_current_timestamp(self):
        # ミリ秒まで含めたタイムスタンプを取得
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        
    def insert_task(self, task):
        try:
            insert_sql = '''
            INSERT INTO tasks (task_name, character_prompt, user_prompt)
            VALUES (?, ?, ?)
            '''
            data = (
                task['task_name'],
                task.get('character_prompt', ''),
                task.get('user_prompt', '')
            )
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
            return self.cursor.lastrowid  # 挿入したタスクのIDを返す
        except sqlite3.IntegrityError as e:
            if 'UNIQUE constraint failed: tasks.task_name' in str(e):
                logging.error(f"タスク名 '{task['task_name']}' は既に存在します。")
                return self.get_task_id(task['task_name'])  # 既存のタスクIDを返す
            else:
                logging.error(f"タスクの挿入中にエラーが発生しました: {e}")
            return None
        except Exception as e:
            logging.error(f"タスクの挿入中にエラーが発生しました: {e}")
            return None

    def get_task_id(self, task_name):
        select_sql = 'SELECT task_id FROM tasks WHERE task_name = ?'
        self.cursor.execute(select_sql, (task_name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def insert_word_info(self, task_id, word, info):
        try:
            insert_sql = '''
            INSERT INTO words_info (task_id, word, info)
            VALUES (?, ?, ?)
            '''
            data = (task_id, word, info)
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
            return self.cursor.lastrowid  # 挿入したワード情報のIDを返す
        except sqlite3.IntegrityError as e:
            return self.get_word_info_id(task_id, word)  # 既存の word_info_id を返す
        except Exception as e:
            logging.error(f"ワード情報の挿入中にエラーが発生しました: {e}")
            return None

    def get_word_info_id(self, task_id, word):
        select_sql = 'SELECT word_info_id FROM words_info WHERE task_id = ? AND word = ?'
        self.cursor.execute(select_sql, (task_id, word))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def insert_conversation(self, entry):
        # token_count を削除し、input_token, output_token を使う
        insert_sql = '''
        INSERT INTO conversations (task_id, word_info_id, talk_num, user, assistant, before_user, before_assistant, input_token, output_token, processing_time, temperature, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            entry['task_id'],
            entry['word_info_id'],
            entry['talk_num'],
            entry['user'],
            entry['assistant'],
            entry['before_user'],
            entry['before_assistant'],
            entry['input_token'],    # 新規追加
            entry['output_token'],   # 新規追加
            entry['processing_time'],
            entry['temperature'],
            entry['created_at']
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.error(f"データベースへの挿入中にエラーが発生しました: {e}")
        except Exception as e:
            logging.error(f"データベースへの挿入中にエラーが発生しました: {e}")

    def check_generated_qa_exists(self, task_name, word_info_id, talk_nums):
        select_sql = '''
        SELECT COUNT(*) FROM generated_qas
        WHERE task_name = ? AND word_info_id = ? AND talk_nums = ?
        '''
        self.cursor.execute(select_sql, (task_name, word_info_id, talk_nums))
        result = self.cursor.fetchone()
        return result[0] > 0

    def insert_generated_qa(self, entry):
        # generated_qasテーブルへの挿入時にtoken_countの代わりにinput_token/output_tokenを使用
        insert_sql = '''
        INSERT INTO generated_qas (task_name, task_id, word_info_id, talk_nums, question, answer, input_token, output_token, processing_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            entry['task_name'],
            entry['task_id'],
            entry['word_info_id'],
            entry['talk_nums'],
            entry['question'],
            entry['answer'],
            entry['input_token'],
            entry['output_token'],
            entry['processing_time'],
            self.get_current_timestamp()
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.error(f"データベースへの挿入中にエラーが発生しました（重複データの可能性）: {e}")
        except Exception as e:
            logging.error(f"データベースへの挿入中にエラーが発生しました: {e}")


    def get_task_name_by_id(self, task_id):
        select_sql = 'SELECT task_name FROM tasks WHERE task_id = ?'
        self.cursor.execute(select_sql, (task_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_conversations_by_task_name(self, task_name):
        select_sql = '''
        SELECT c.conversation_id, c.task_id, c.user, c.assistant
        FROM conversations c
        INNER JOIN tasks t ON c.task_id = t.task_id
        WHERE t.task_name = ?
        '''
        self.cursor.execute(select_sql, (task_name,))
        rows = self.cursor.fetchall()
        conversations = []
        for row in rows:
            conversations.append({
                'conversation_id': row[0],
                'task_id': row[1],
                'content': f"User: {row[2]}\nAssistant: {row[3]}"
            })
        return conversations

    def get_conversations_with_task_name(self, task_name):
        select_sql = '''
        SELECT c.task_id, c.word_info_id, c.talk_num, c.user, c.assistant ,c.before_user, c.before_assistant, t.task_name, wi.word, t.character_prompt, t.user_prompt
        FROM conversations c
        INNER JOIN tasks t ON c.task_id = t.task_id
        INNER JOIN words_info wi ON c.word_info_id = wi.word_info_id
        WHERE t.task_name = ?
        ORDER BY wi.word, CAST(c.talk_num AS INTEGER)
        '''
        try:
            self.cursor.execute(select_sql, (task_name,))
            rows = self.cursor.fetchall()
            conversations = []
            for row in rows:
                conversations.append({
                    'task_id': row[0],
                    'word_info_id': row[1],
                    'talk_num': row[2],
                    'user': row[3],
                    'assistant': row[4],
                    'before_user': row[5],
                    'before_assistant': row[6],
                    'task_name': row[7],
                    'word': row[8],
                    'character_prompt': row[9],
                    'user_prompt': row[10]
                })
            return conversations
        except Exception as e:
            logging.error(f"データベースからのデータ取得中にエラーが発生しました: {e}")
            return []
        

    def get_generated_qas_by_task_name(self, task_name):
        select_sql = '''
        SELECT gqa.talk_nums, gqa.task_name, wi.word, gqa.question, gqa.answer, gqa.input_token, gqa.output_token, gqa.processing_time
        FROM generated_qas gqa
        INNER JOIN words_info wi ON gqa.word_info_id = wi.word_info_id
        WHERE gqa.task_name = ?
        '''
        try:
            self.cursor.execute(select_sql, (task_name,))
            rows = self.cursor.fetchall()
            qas_list = []
            for row in rows:
                qas_list.append({
                    'talk_nums': row[0],
                    'task_name': row[1],
                    'word': row[2],
                    'question': row[3],
                    'answer': row[4],
                    'input_token': row[5],
                    'output_token': row[6],
                    'processing_time': row[7]
                })
            return qas_list
        except Exception as e:
            logging.error(f"データベースからのデータ取得中にエラーが発生しました: {e}")
            return []
        

    def get_conversation_chunks_by_task_name(self, task_name):
        select_sql = '''
        SELECT c.task_id, c.word_info_id, c.talk_num, c.user, c.assistant, t.task_name, wi.word, t.character_prompt, t.user_prompt
        FROM conversations c
        INNER JOIN tasks t ON c.task_id = t.task_id
        INNER JOIN words_info wi ON c.word_info_id = wi.word_info_id
        WHERE t.task_name = ?
        ORDER BY wi.word, CAST(c.talk_num AS INTEGER)
        '''
        try:
            self.cursor.execute(select_sql, (task_name,))
            rows = self.cursor.fetchall()
            conversations = []
            for row in rows:
                conversations.append({
                    'task_id': row[0],
                    'word_info_id': row[1],
                    'talk_num': row[2],
                    'user': row[3],
                    'assistant': row[4],
                    'task_name': row[5],
                    'word': row[6],
                    'character_prompt': row[7],
                    'user_prompt': row[8]
                })
            return conversations
        except Exception as e:
            logging.error(f"データベースからのデータ取得中にエラーが発生しました: {e}")
            return []


    def has_generated_qas(self, task_name: str) -> bool:
        """
        指定された task_name に対して、generated_qas テーブルに既に Q&A が存在するかを確認します。
        """
        select_sql = 'SELECT COUNT(*) FROM generated_qas WHERE task_name = ?'
        self.cursor.execute(select_sql, (task_name,))
        result = self.cursor.fetchone()
        return result[0] > 0
    

    def insert_evaluated_answer(self, result_entry):
        insert_sql = '''
        INSERT INTO evaluated_answers (
            qa_id, task_name, task_id, word_info_id, talk_nums, question,
            expected_answer, gpt_response, get_context, get_talk_nums,
            input_token, output_token,     -- token_count の代わり
            processing_time, model, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        data = (
            result_entry['qa_id'],
            result_entry['task_name'],
            result_entry['task_id'],
            result_entry['word_info_id'],
            result_entry['talk_nums'],
            result_entry['question'],
            result_entry['expected_answer'],
            result_entry['gpt_response'],
            result_entry['get_context'],
            result_entry['get_talk_nums'],
            result_entry['input_token'],   # 新規追加
            result_entry['output_token'],  # 新規追加
            result_entry['processing_time'],
            result_entry['model'],
            result_entry.get('created_at', self.get_current_timestamp())
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise DataAlreadyExistsError(f"評価結果は既に存在します。詳細: {e}")
        except Exception as e:
            logging.error(f"評価結果の保存中にエラーが発生しました: {e}")
            raise e
        
    def get_generated_qas_with_ids_by_task_name(self, task_name):
        select_sql = '''
        SELECT
            gqa.qa_id,
            gqa.talk_nums,
            gqa.task_name,
            wi.word,
            gqa.question,
            gqa.answer,
            gqa.input_token,
            gqa.output_token,
            gqa.processing_time,
            gqa.task_id,
            gqa.word_info_id
        FROM generated_qas gqa
        INNER JOIN words_info wi ON gqa.word_info_id = wi.word_info_id
        WHERE gqa.task_name = ?
        '''
        try:
            self.cursor.execute(select_sql, (task_name,))
            rows = self.cursor.fetchall()
            qas_list = []
            for row in rows:
                qas_list.append({
                    'qa_id': row[0],
                    'talk_nums': row[1],
                    'task_name': row[2],
                    'word': row[3],
                    'question': row[4],
                    'answer': row[5],
                    'input_token': row[6],
                    'output_token': row[7],
                    'processing_time': row[8],
                    'task_id': row[9],
                    'word_info_id': row[10]
                })
            return qas_list
        except Exception as e:
            logging.error(f"データベースからのデータ取得中にエラーが発生しました: {e}")
            return []
        
    def has_evaluated_answers(self, task_name, model_name):
        """
        指定した task_name と model_name の組み合わせで既に評価済みの回答があるかどうかを返す
        """
        query = """
            SELECT COUNT(*) 
            FROM evaluated_answers 
            WHERE task_name = ? AND model = ?
        """
        self.cursor.execute(query, (task_name, model_name))
        count = self.cursor.fetchone()[0]

        return count > 0
    

    def insert_rag_result(self, result_entry):
        """
        RAGモデルの結果を rag_results テーブルに挿入します。
        """
        insert_sql = '''
        INSERT INTO rag_results (
            qa_id, task_name, task_id,
            talk_num_1, talk_num_2, talk_num_3, talk_num_4, talk_num_5,
            cosine_similarity_1, cosine_similarity_2, cosine_similarity_3, cosine_similarity_4, cosine_similarity_5,
            BM25_score_1, BM25_score_2, BM25_score_3, BM25_score_4, BM25_score_5,
            rss_rank_1, rss_rank_2, rss_rank_3, rss_rank_4, rss_rank_5,
            rerank_score_1, rerank_score_2, rerank_score_3, rerank_score_4, rerank_score_5,
            processing_time, model, input_token_count, output_token_count, subject, created_at
        ) VALUES (?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?)
        '''
        data = (
            result_entry['qa_id'],
            result_entry['task_name'],
            result_entry['task_id'],
            result_entry.get('talk_num_1', ''),
            result_entry.get('talk_num_2', ''),
            result_entry.get('talk_num_3', ''),
            result_entry.get('talk_num_4', ''),
            result_entry.get('talk_num_5', ''),
            result_entry.get('cosine_similarity_1', 0.0),
            result_entry.get('cosine_similarity_2', 0.0),
            result_entry.get('cosine_similarity_3', 0.0),
            result_entry.get('cosine_similarity_4', 0.0),
            result_entry.get('cosine_similarity_5', 0.0),
            result_entry.get('BM25_score_1', None),
            result_entry.get('BM25_score_2', None),
            result_entry.get('BM25_score_3', None),
            result_entry.get('BM25_score_4', None),
            result_entry.get('BM25_score_5', None),
            result_entry.get('rss_rank_1', None),
            result_entry.get('rss_rank_2', None),
            result_entry.get('rss_rank_3', None),
            result_entry.get('rss_rank_4', None),
            result_entry.get('rss_rank_5', None),
            result_entry.get('rerank_score_1', None),
            result_entry.get('rerank_score_2', None),
            result_entry.get('rerank_score_3', None),
            result_entry.get('rerank_score_4', None),
            result_entry.get('rerank_score_5', None),
            result_entry.get('processing_time', None),
            result_entry.get('model', ''),
            result_entry.get('input_token_count', 0),    # 主語抽出時などのinput token数
            result_entry.get('output_token_count', 0),    # 主語抽出時などのoutput token数
            result_entry.get('subject', ''),  # 主語を保存
            result_entry.get('created_at', self.get_current_timestamp())
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.error(f"rag_results テーブルへの挿入中にエラーが発生しました: {e}")
            raise DataAlreadyExistsError(f"RAG結果は既に存在します。詳細: {e}")
        except Exception as e:
            logging.error(f"rag_results テーブルへの挿入中にエラーが発生しました: {e}")
            raise e

    def get_qa_id(self, task_name, word_info_id, talk_nums):
        """
        generated_qas テーブルから qa_id を取得します。
        """
        select_sql = '''
        SELECT qa_id FROM generated_qas
        WHERE task_name = ? AND word_info_id = ? AND talk_nums = ?
        '''
        try:
            self.cursor.execute(select_sql, (task_name, word_info_id, talk_nums))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logging.error(f"qa_id の取得中にエラーが発生しました: {e}")
            return None

    def get_rag_results_by_task_name(self, task_name):
        """
        指定されたタスク名に関連する RAG結果を取得します。
        """
        select_sql = '''
        SELECT * FROM rag_results WHERE task_name = ? ORDER BY rag_id;
        '''
        try:
            self.cursor.execute(select_sql, (task_name,))
            rows = self.cursor.fetchall()
            # カラム名を取得
            column_names = [description[0] for description in self.cursor.description]
            results = []
            for row in rows:
                result_entry = dict(zip(column_names, row))
                results.append(result_entry)
            return results
        except Exception as e:
            logging.error(f"rag_results テーブルからのデータ取得中にエラーが発生しました: {e}")
            return []
        
    def get_conversation_history(self, task_name, talk_num_list):
        select_sql = '''
        SELECT c.user, c.assistant, c.created_at
        FROM conversations c
        INNER JOIN tasks t ON c.task_id = t.task_id
        WHERE t.task_name = ? AND c.talk_num IN ({seq})
        ORDER BY c.created_at ASC
        '''.format(seq=','.join(['?']*len(talk_num_list)))

        params = [task_name] + talk_num_list

        try:
            self.cursor.execute(select_sql, params)
            rows = self.cursor.fetchall()
            conversation_history = []
            for row in rows:
                conversation_history.append({
                    'user': row[0],
                    'assistant': row[1],
                    'created_at': row[2]
                })
            return conversation_history
        except Exception as e:
            logging.error(f"会話履歴の取得中にエラーが発生しました: {e}")
            return []

    def close(self):
        self.conn.close()
