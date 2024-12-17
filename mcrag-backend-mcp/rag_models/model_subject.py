# rag_models/model_subject.py

from typing import Any, Dict
from .base_model import BaseRAGModel
from database import ConversationDatabase
from vector_db import VectorDatabase
from call_gpt import GPTClient
import numpy as np
import constants as c
import time


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
        self.subject_extraction_input_tokens = 0
        self.subject_extraction_output_tokens = 0   


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
        # トークン数取得(usageがあると仮定)
        if 'usage' in response:
            self.subject_extraction_input_tokens = response['usage'].get('prompt_tokens', 0)
            self.subject_extraction_output_tokens = response['usage'].get('completion_tokens', 0)
        else:
            self.subject_extraction_input_tokens = 0
            self.subject_extraction_output_tokens = 0
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
        start_time = time.time()
        # 1. 主語抽出
        subject = self.extract_subject(query)

        # 主語がなければ空のコンテキスト返す
        if not subject:
            return {
                'get_context_1': '',
                'get_context_2': '',
                'get_context_3': '',
                'get_context_4': '',
                'get_context_5': '',
                'get_talk_nums': ''
            }

        # 2. 会話履歴取得
        conversations = self.database.get_conversations_with_task_name(task_name)
        # conversationsは [{task_id, word_info_id, talk_num, user, assistant, ...}, ...]

        # (task_id, word_info_id) ごとにグループ化
        grouped_conversations = {}
        for convo in conversations:
            key = (convo['task_id'], convo['word_info_id'])
            if key not in grouped_conversations:
                grouped_conversations[key] = []
            grouped_conversations[key].append(convo)

        # 3. subject含む発話を起点に5件のチャンクを作る
        chunks = []
        for key, convos in grouped_conversations.items():
            # talk_numでソート
            convos_sorted = sorted(convos, key=lambda x: int(x['talk_num']))

            for i, line in enumerate(convos_sorted):
                text_line = f"User: {line['user']}\nAssistant: {line['assistant']}"
                if subject in line['user'] or subject in line['assistant']:
                    # i番目から最大5行とる
                    chunk_lines = convos_sorted[i:i+5]
                    if len(chunk_lines) > 0:
                        talk_nums_list = [ch['talk_num'] for ch in chunk_lines]
                        # テキストコンテキスト(ユーザ+アシスタント)も返却用にまとめる
                        chunk_texts = [f"User: {ch['user']}\nAssistant: {ch['assistant']}" for ch in chunk_lines]
                        chunk_full_text = "\n".join(chunk_texts)
                        chunks.append((talk_nums_list, chunk_full_text))

        if not chunks:
            # 該当会話なし
            end_time = time.time()
            processing_time = end_time - start_time
            # RAG結果保存(該当なしの場合はtalk_nums空)
            task_id = self.database.get_task_id(task_name)
            result_entry = {
                'qa_id': qa_id,
                'task_name': task_name,
                'task_id': task_id,
                'talk_num_1': '',
                'talk_num_2': '',
                'talk_num_3': '',
                'talk_num_4': '',
                'talk_num_5': '',
                'cosine_similarity_1': 0.0,
                'cosine_similarity_2': 0.0,
                'cosine_similarity_3': 0.0,
                'cosine_similarity_4': 0.0,
                'cosine_similarity_5': 0.0,
                'processing_time': processing_time,
                'model': c.SUBJECT_SEARCH,
                'input_token_count': self.subject_extraction_input_tokens,
                'output_token_count': self.subject_extraction_output_tokens,
                'subject': subject,
                'created_at': self.database.get_current_timestamp()
            }
            self.database.insert_rag_result(result_entry)
            return {
                'get_context_1': f"抽出した主語: {subject}\n(該当会話なし)",
                'get_context_2': '',
                'get_context_3': '',
                'get_context_4': '',
                'get_context_5': '',
                'get_talk_nums': ''
            }

        # 4. クエリベクトルを取得
        query_vector = self.vector_db.compute_query_embedding(query)

        # タスク全体の talk_num -> vector のマッピングを作成
        # fetch_vectors_and_contents で全ての会話ベクトルを取得
        vectors, contents, vector_ids, talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)

        # talk_numは文字列、vector_idsはint、順番が対応しているので talk_num -> vector を作成
        talk_num_to_vector = {}
        for i, tn in enumerate(talk_nums):
            talk_num_to_vector[tn] = vectors[i]

        best_similarity = -1.0
        best_chunk_text = None
        best_talk_nums = []

        # 5. 各チャンクについて、5つの talk_num のベクトルを平均する
        for talk_nums_list, chunk_full_text in chunks:
            chunk_vectors = []
            for tn in talk_nums_list:
                if tn in talk_num_to_vector:
                    chunk_vectors.append(talk_num_to_vector[tn])
            
            if not chunk_vectors:
                continue

            # 平均ベクトル計算
            chunk_vector = np.mean(chunk_vectors, axis=0).astype(np.float32)

            # 6. 類似度計算
            similarity = self.vector_db.cosine_similarity(query_vector, chunk_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk_text = chunk_full_text
                best_talk_nums = talk_nums_list

        # best_chunk_textを行ごとに分割し、最大5行取り出す
        lines = best_chunk_text.split("\n")
        contexts = lines[:5]
        while len(contexts) < 5:
            contexts.append('')

        get_talk_nums = ','.join(str(num) for num in best_talk_nums)

        end_time = time.time()
        processing_time = end_time - start_time

        # RAG結果の保存
        task_id = self.database.get_task_id(task_name)
        # best_talk_numsには最大5件のtalk_numが入る
        # 不足分は空で埋める
        tnums = best_talk_nums + ['']*(5 - len(best_talk_nums))
        result_entry = {
            'qa_id': qa_id,
            'task_name': task_name,
            'task_id': task_id,
            'talk_num_1': tnums[0],
            'talk_num_2': tnums[1],
            'talk_num_3': tnums[2],
            'talk_num_4': tnums[3],
            'talk_num_5': tnums[4],
            # 今回はbest chunkのみなので cosine_similarity_1 にbestを、他は0.0
            'cosine_similarity_1': best_similarity,
            'cosine_similarity_2': 0.0,
            'cosine_similarity_3': 0.0,
            'cosine_similarity_4': 0.0,
            'cosine_similarity_5': 0.0,
            'processing_time': processing_time,
            'model': c.SUBJECT_SEARCH,
            'input_token_count': self.subject_extraction_input_tokens,
            'output_token_count': self.subject_extraction_output_tokens,
            'subject': subject,
            'created_at': self.database.get_current_timestamp()
        }
        self.database.insert_rag_result(result_entry)

        return {
            'get_context_1': contexts[0],
            'get_context_2': contexts[1],
            'get_context_3': contexts[2],
            'get_context_4': contexts[3],
            'get_context_5': contexts[4],
            'get_talk_nums': get_talk_nums
        }