# rag_models/model_subject.py

from typing import Any, Dict
from .base_model import BaseRAGModel
from database import ConversationDatabase
from vector_db import VectorDatabase
from call_gpt import GPTClient

class SubjectSearchModel(BaseRAGModel):
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db'):
        super().__init__(api_key=api_key, db_file=db_file)
        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

        # Function Callingで使用するFunction定義
        self.functions = [
            {
                "name": "extract_subject",
                "description": "Extracts the proper nouns contained in the subject from the given Japanese query. Only that single word is returned without any additional description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The extracted subject as a single noun from the query. Must be exactly one word from the query."
                        }
                    },
                    "required": ["subject"]
                }
            }
        ]

    def extract_subject(self, query: str) -> str:
        """
        LLMのFunction Callingを用いて、クエリから主語となる単語を一語抽出する。
        ポリシー：
        - クエリから主語として最適な名詞を一語だけ選ぶ
        - 返す値はその名詞一語のみ。余計な文字や引用符、補足はいらない
        - クエリに主語らしき名詞がない場合は空文字を返す
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a function to extract the main subject from a Japanese query. "
                    "Rules:\n"
                    "1. Identify the main subject noun (a single word) from the user's query.\n"
                    "2. Return only that one word. No other text, no explanation.\n"
                    "3. If multiple candidates exist, choose the noun that best represents the main topic.\n"
                    "4. If no suitable subject noun is found, return an empty string.\n"
                    "5. Do not add quotes or extra punctuation around the word.\n"
                )
            },
            {
                "role": "user",
                "content": f"以下のクエリから主語となる固有名詞を一語抽出してください:\n{query}"
            }
        ]

        # function_callで特定の関数呼び出しを強制
        function_call = {"name": "extract_subject"}

        response = self.gpt_client.call_gpt_function(
            messages=messages,
            functions=self.functions,
            function_call=function_call,
            temperature=0.0
        )

        # responseからfunction callの引数を取得
        if response and "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "function_call" in choice["message"]:
                # 引数はJSON文字列として返されるのでパース
                args_str = choice["message"]["function_call"]["arguments"]
                import json
                args = json.loads(args_str)
                subject = args.get("subject", "")
                return subject

        return ""

    def retrieve_context(self, query: str, task_name: str, qa_id: int) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストを取得します。
        ここでは LLM Function Callingで抽出した主語をコンテキストとして返します。
        """
        # LLMにより主語抽出
        subject = self.extract_subject(query)

        return {
            'get_context_1': f"抽出した主語: {subject}",
            'get_context_2': '',
            'get_context_3': '',
            'get_context_4': '',
            'get_context_5': '',
            'get_talk_nums': ''
        }
