# rag_models/base_model.py

from abc import ABC, abstractmethod

class BaseRAGModel(ABC):
    @abstractmethod
    def retrieve_context(self, query: str, task_name: str) -> str:
        """会話履歴からクエリに関連するコンテキストを取得する"""
        pass