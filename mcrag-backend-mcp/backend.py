import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from conversation_generator import ConversationGenerator
from question_generator import QuestionGenerator
from answer_evaluator import AnswerEvaluator
from typing import List, Dict, Any
from fastapi.responses import JSONResponse




app = FastAPI()


@app.post("/generate_conversation")
async def generate_conversation(request: Request):
    try:
        # リクエストボディを取得
        data = await request.json()

        # ConversationGenerator を呼び出す
        generator = ConversationGenerator(
            temperature=data["temperature"],
            api_key=data["api_key"],
            task_name=data["task_name"],
            words_info=data["words_info"],
            character_prompt=data["character_prompt"],
            user_prompt=data["user_prompt"]
        )

        # 会話生成の実行
        conversations = generator.run_conversation()

        # 結果の返却
        return JSONResponse({
            "conversations": conversations,
            "total_tokens": generator.total_tokens,
            "total_processing_time": generator.total_processing_time
        })
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500)
    

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