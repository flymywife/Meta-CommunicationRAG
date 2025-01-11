# rag_models/model_topic_word.py

from typing import Any, Dict
from .base_model import BaseRAGModel
from database import ConversationDatabase
from vector_db import VectorDatabase
from call_gpt import GPTClient
import numpy as np
import constants as c
import time
import json


class TopicWordSearchModel(BaseRAGModel):
    def __init__(self, api_key: str, db_file: str = 'conversation_data.db'):
        super().__init__(api_key=api_key, db_file=db_file)
        # GPTClientのインスタンスを作成
        self.gpt_client = GPTClient(api_key=self.api_key)

        # Function Calling で使用するFunction定義
        self.functions = [
            {
                "name": "extract_main_proper_noun",
                "description": (
                    "Extract from the given Japanese query the single proper noun that best represents "
                    "the main topic or entity the user is asking about. If none is relevant, return an empty string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "proper_noun": {
                            "type": "string",
                            "description": (
                                "The single proper noun (one word) that is the main focus of the user's query. "
                                "No extra text or quotes."
                            )
                        }
                    },
                    "required": ["proper_noun"]
                }
            }
        ]

        self.subject_extraction_input_tokens = 0
        self.subject_extraction_output_tokens = 0

    def extract_subject(self, query: str) -> str:
        """
        LLMのFunction Callingを用いて、クエリから
        「ユーザが本当に知りたい（尋ねている）対象となる固有名詞」を一語だけ抽出する。

        ポリシー：
        - ユーザの呼びかけ先や相手の名前(例: 「ロボくん」など)ではなく、
          ユーザが情報を求めている「本題となる固有名詞」を選ぶ
        - 返す値はその固有名詞一語のみ。余計な文字や引用符・補足は不要
        - クエリに適切な固有名詞がない場合は空文字を返す
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a function that extracts from a Japanese query the single proper noun (one word) that "
                    "the user is actually inquiring about. Ignore any name used purely for addressing or calling out. "
                    "If multiple proper nouns appear, choose the one that is truly the focus of the question. "
                    "If no appropriate proper noun is found, return an empty string. "
                    "Output only the single noun, no quotes or explanations."
                )
            },
            {
                "role": "user",
                "content": f"以下のクエリから、ユーザが本当に知りたい情報の対象となる固有名詞を一語抽出してください:\n{query}"
            }
        ]

        # function_callで特定の関数呼び出しを強制
        function_call = {"name": "extract_main_proper_noun"}

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

        # response から function_call の引数を取得
        if response and "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "function_call" in choice["message"]:
                args_str = choice["message"]["function_call"]["arguments"]
                try:
                    args = json.loads(args_str)
                    proper_noun = args.get("proper_noun", "")
                    return proper_noun
                except:
                    return ""
        return ""

    def retrieve_context(self, query: str, task_name: str, qa_id: int) -> Dict[str, Any]:
        """
        クエリに関連するコンテキストを取得します。
        1. LLM Function Callingで抽出した固有名詞(本題)を取得
        2. その固有名詞をキーに会話履歴を探索して最適なチャンクをピックアップ
        3. ピックアップしたチャンクを返す
        """
        start_time = time.time()

        # 1. 固有名詞(本題)を抽出
        subject = self.extract_subject(query)

        # もし見つからなければ空コンテキスト返す
        if not subject:
            end_time = time.time()
            processing_time = end_time - start_time
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
                'model': c.TOPIC_WORD_SEARCH,
                'input_token_count': self.subject_extraction_input_tokens,
                'output_token_count': self.subject_extraction_output_tokens,
                'subject': subject,
                'created_at': self.database.get_current_timestamp()
            }
            self.database.insert_rag_result(result_entry)

            return {
                'get_context_1': '',
                'get_context_2': '',
                'get_context_3': '',
                'get_context_4': '',
                'get_context_5': '',
                'get_talk_nums': ''
            }

        # 2. 会話履歴を取得
        conversations = self.database.get_conversations_with_task_name(task_name)
        # conversations -> [{task_id, word_info_id, talk_num, user, assistant, ...}, ...]

        # (task_id, word_info_id) ごとにグループ化
        grouped_conversations = {}
        for convo in conversations:
            t_id = convo['task_id']
            if t_id not in grouped_conversations:
                grouped_conversations[t_id] = []
            grouped_conversations[t_id].append(convo)

        # subjectを含む発話を起点に最大5件のチャンクを作る
        chunks = []
        for key, convos in grouped_conversations.items():
            convos_sorted = sorted(convos, key=lambda x: int(x['talk_num']))

            for i, line in enumerate(convos_sorted):
                text_line = f"User: {line['user']}\nAssistant: {line['assistant']}"
                # subject を含むかどうか
                if subject in line['user'] or subject in line['assistant']:
                    chunk_lines = convos_sorted[i:i+5]
                    if len(chunk_lines) > 0:
                        talk_nums_list = [ch['talk_num'] for ch in chunk_lines]
                        chunk_texts = [
                            f"User: {ch['user']}\nAssistant: {ch['assistant']}"
                            for ch in chunk_lines
                        ]
                        chunk_full_text = "\n".join(chunk_texts)
                        chunks.append((talk_nums_list, chunk_full_text))

        # 該当チャンクがなければ空を返す
        if not chunks:
            end_time = time.time()
            processing_time = end_time - start_time
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
                'model': c.TOPIC_WORD_SEARCH,
                'input_token_count': self.subject_extraction_input_tokens,
                'output_token_count': self.subject_extraction_output_tokens,
                'subject': subject,
                'created_at': self.database.get_current_timestamp()
            }
            self.database.insert_rag_result(result_entry)

            return {
                'get_context_1': f"抽出した固有名詞: {subject}\n(該当会話なし)",
                'get_context_2': '',
                'get_context_3': '',
                'get_context_4': '',
                'get_context_5': '',
                'get_talk_nums': ''
            }

        # 3. Embedding でさらに最適チャンクを選ぶ
        query_vector = self.vector_db.compute_query_embedding(query)

        # タスク全体の talk_num -> vector / content マッピング取得
        vectors, contents, vector_ids, all_talk_nums = self.vector_db.fetch_vectors_and_contents(task_name)
        talk_num_to_vector = {}
        talk_num_to_content = {}
        for i, tn in enumerate(all_talk_nums):
            talk_num_to_vector[tn] = vectors[i]
            talk_num_to_content[tn] = contents[i]

        best_similarity = -1.0
        best_chunk_text = None
        best_talk_nums = []

        for talk_nums_list, chunk_full_text in chunks:
            chunk_vectors = []
            for tn in talk_nums_list:
                if tn in talk_num_to_vector:
                    chunk_vectors.append(talk_num_to_vector[tn])
            if not chunk_vectors:
                continue

            chunk_vector = np.mean(chunk_vectors, axis=0).astype(np.float32)
            similarity = self.vector_db.cosine_similarity(query_vector, chunk_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk_text = chunk_full_text
                best_talk_nums = talk_nums_list

        end_time = time.time()
        processing_time = end_time - start_time

        # best_talk_nums に対応する content を最大5件取得
        selected_contents = [talk_num_to_content.get(tn, '') for tn in best_talk_nums]
        while len(selected_contents) < 5:
            selected_contents.append('')
        get_talk_nums = ','.join(str(num) for num in best_talk_nums)

        # DB保存
        task_id = self.database.get_task_id(task_name)
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
            'cosine_similarity_1': best_similarity,
            'cosine_similarity_2': 0.0,
            'cosine_similarity_3': 0.0,
            'cosine_similarity_4': 0.0,
            'cosine_similarity_5': 0.0,
            'processing_time': processing_time,
            'model': c.TOPIC_WORD_SEARCH,
            'input_token_count': self.subject_extraction_input_tokens,
            'output_token_count': self.subject_extraction_output_tokens,
            'subject': subject,
            'created_at': self.database.get_current_timestamp()
        }
        self.database.insert_rag_result(result_entry)

        return {
            'get_context_1': selected_contents[0],
            'get_context_2': selected_contents[1],
            'get_context_3': selected_contents[2],
            'get_context_4': selected_contents[3],
            'get_context_5': selected_contents[4],
            'get_talk_nums': get_talk_nums
        }
