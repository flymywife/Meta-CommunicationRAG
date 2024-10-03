import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from conversation_generator import ConversationGenerator
from question_generator import QuestionGenerator
from answer_evaluator import AnswerEvaluator
from typing import List, Dict



app = FastAPI()

# リクエストボディのモデル定義
class ConversationRequest(BaseModel):
    temperature: float
    api_key: str
    task_name: str
    words_info: List[Dict[str, str]]
    num_turns_per_word: int  # 変数名を変更
    character_prompt: str
    user_prompt: str

# レスポンスボディのモデル定義
class ConversationResponse(BaseModel):
    conversations: List[Dict[str, str]]
    total_tokens: int
    total_processing_time: float

@app.post("/generate_conversation", response_model=ConversationResponse)
async def generate_conversation(request: ConversationRequest):
    try:
        generator = ConversationGenerator(
            temperature=request.temperature,
            api_key=request.api_key,
            task_name=request.task_name,
            words_info=request.words_info,
            character_prompt=request.character_prompt,
            user_prompt=request.user_prompt
        )
        conversations = generator.run_conversation(
            num_turns_per_word=request.num_turns_per_word
        )
        total_tokens = generator.total_tokens
        total_processing_time = generator.total_processing_time

        # データベース接続をクローズ
        generator.close()

        # 必要な情報を返す
        return {
            "conversations": conversations,
            "total_tokens": total_tokens,
            "total_processing_time": total_processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

# リクエストボディのモデル定義
class QuestionRequest(BaseModel):
    temperature: float
    api_key: str
    json_data: Dict  # JSONデータを辞書として受け取る

# レスポンスボディのモデル定義
class QuestionResponse(BaseModel):
    results: List[Dict[str, str]]
    total_tokens: int
    total_processing_time: float

@app.post("/generate_questions", response_model=QuestionResponse)
async def generate_questions(request: QuestionRequest):
    try:
        question_generator = QuestionGenerator(
            temperature=request.temperature,
            api_key=request.api_key
        )

        # JSONデータから会話チャンクを取得
        conversation_chunks = question_generator.parse_json_data(request.json_data)

        # 質問と回答の生成
        results = question_generator.generate_question_and_answer(conversation_chunks)

        return {
            "results": results,
            "total_tokens": question_generator.total_tokens,
            "total_processing_time": question_generator.total_processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# リクエストボディのモデル定義
class EvaluationRequest(BaseModel):
    temperature: float
    api_key: str
    # CSVファイルの内容を文字列として受け取る
    csv_content: str  # CSVの文字列

# レスポンスボディのモデル定義
class EvaluationResponse(BaseModel):
    results: list
    total_tokens: int
    total_processing_time: float

@app.post("/evaluate_answers", response_model=EvaluationResponse)
async def evaluate_answers(request: EvaluationRequest):
    try:
        answer_evaluator = AnswerEvaluator(
            temperature=request.temperature,
            api_key=request.api_key
        )
        # CSVファイルの内容を DataFrame に変換
        from io import StringIO
        csv_file = StringIO(request.csv_content)
        df = pd.read_csv(csv_file)
        # 必要な列が存在するか確認
        required_columns = ['talk_nums', 'task_name', 'word', 'query', 'answer']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="CSVに必要な列がありません。")

        # 回答の評価
        results = answer_evaluator.evaluate_answers(df)
        return {
            "results": results,
            "total_tokens": answer_evaluator.total_tokens,
            "total_processing_time": answer_evaluator.total_processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))