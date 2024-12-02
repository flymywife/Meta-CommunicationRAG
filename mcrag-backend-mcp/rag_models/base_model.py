# rag_models/base_model.py

from abc import ABC, abstractmethod
import openai
import numpy as np
from typing import Any,Dict, List
import constants as c
from vector_db import VectorDatabase  # vector_db.py をインポート
from database import ConversationDatabase # データベースクラスをインポート




class BaseRAGModel(ABC):
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db'):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.db_file = db_file

        # VectorDatabase のインスタンスを作成
        self.vector_db = VectorDatabase(api_key=self.api_key, db_file=self.db_file)
        self.database = ConversationDatabase(db_file=self.db_file)


    @abstractmethod
    def retrieve_context(self,qa_id: int , query: str, task_name: str) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストを取得し、指定された形式の辞書を返します。

        Returns:
            {
                'get_context_1': str,
                'get_context_2': str,
                'get_context_3': str,
                'get_context_4': str,
                'get_context_5': str,
                'get_talk_nums': str
            }
        """
        pass

    def close(self):
        """
        データベース接続を閉じます。
        """
        self.vector_db.close()
