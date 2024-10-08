import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from conversation_generator import ConversationGenerator
from question_generator import QuestionGenerator
from answer_evaluator import AnswerEvaluator
from typing import List, Dict, Any
from fastapi.responses import JSONResponse
import logging




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
        logging.error(f"Error in /generate_conversation: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)
    



@app.post("/generate_questions")
async def generate_questions(request: Request):
    try:
        # リクエストボディを取得
        data = await request.json()

        # 必要なフィールドが存在するかチェック
        required_fields = ["temperature", "api_key", "json_data"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        temperature = data["temperature"]
        api_key = data["api_key"]
        json_data = data["json_data"]

        # QuestionGenerator を呼び出す
        question_generator = QuestionGenerator(
            temperature=temperature,
            api_key=api_key
        )

        # JSONデータから会話チャンクを取得
        conversation_chunks = question_generator.parse_json_data(json_data)

        # 質問と回答の生成
        results = question_generator.generate_question_and_answer(conversation_chunks)

        # レスポンスの準備
        response_data = {
            "results": results,
            "total_tokens": question_generator.total_tokens,
            "total_processing_time": question_generator.total_processing_time
        }

        return JSONResponse(content=response_data, status_code=200)

    except HTTPException as he:
        # 既に定義されたHTTPExceptionをそのまま返す
        raise he
    except Exception as e:
        logging.error(f"Error in /generate_questions: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)



@app.post("/evaluate_answers")
async def evaluate_answers(request: Request):
    try:
        data = await request.json()
        logging.info(f"Received data for evaluate_answers: {data}")

        required_fields = ["temperature", "api_key", "json_data"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        temperature = data["temperature"]
        api_key = data["api_key"]
        json_data = data["json_data"]

        # JSONデータがリスト形式かどうかチェック
        if not isinstance(json_data, list):
            raise HTTPException(status_code=422, detail="json_data must be a list of dictionaries.")

        # 必要なキーが存在するか確認
        required_keys = ['talk_nums', 'task_name', 'word', 'query', 'answer']
        for entry in json_data:
            if not all(key in entry for key in required_keys):
                raise HTTPException(status_code=400, detail="Each entry in json_data must contain all required keys.")

        answer_evaluator = AnswerEvaluator(
            temperature=temperature,
            api_key=api_key
        )

        results = answer_evaluator.evaluate_answers(json_data)

        response_data = {
            "results": results,
            "total_tokens": answer_evaluator.total_tokens,
            "total_processing_time": answer_evaluator.total_processing_time
        }

        return JSONResponse(content=response_data, status_code=200)

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in /evaluate_answers: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)
