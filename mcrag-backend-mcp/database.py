# database.py

import sqlite3
import logging
from datetime import datetime

# ログの基本設定
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

class DataAlreadyExistsError(Exception):
    """データが既に存在する場合の例外クラス"""
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
            token_count TEXT,
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
            token_count INTEGER,
            processing_time REAL,
            created_at TEXT,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id),
            FOREIGN KEY (word_info_id) REFERENCES words_info (word_info_id),
            UNIQUE (task_name, word_info_id, talk_nums)  -- 修正後の UNIQUE 制約
)
        ''')
        self.conn.commit()

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
        insert_sql = '''
        INSERT INTO conversations (task_id, word_info_id, talk_num, user, assistant, token_count, processing_time, temperature, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            entry['task_id'],
            entry['word_info_id'],
            entry['talk_num'],
            entry['user'],
            entry['assistant'],
            entry['token_count'],
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
        task_name = entry['task_name']

        insert_sql = '''
        INSERT INTO generated_qas (task_name, task_id, word_info_id, talk_nums, question, answer, token_count, processing_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        data = (
            task_name,
            entry['task_id'],
            entry['word_info_id'],
            entry['talk_nums'],
            entry['question'],
            entry['answer'],
            entry['token_count'],
            entry['processing_time'],
            self.get_current_timestamp()
        )
        try:
            self.cursor.execute(insert_sql, data)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logging.error(f"データベースへの挿入中にエラーが発生しました（重複データの可能性）: {e}")
            # 必要に応じて例外を発生させる
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

    def get_generated_qas_by_task_name(self, task_name):
        select_sql = '''
        SELECT gqa.talk_nums, gqa.task_name, wi.word, gqa.question, gqa.answer, gqa.token_count, gqa.processing_time
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
                    'query': row[3],
                    'answer': row[4],
                    'token_count': row[5],
                    'processing_time': row[6]
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
    

    def close(self):
        self.conn.close()
