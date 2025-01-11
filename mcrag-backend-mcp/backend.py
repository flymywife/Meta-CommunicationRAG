import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from conversation_generator import ConversationGenerator
from question_generator import QuestionGenerator, DataNotFoundError  # DataNotFoundError をインポート
from answer_evaluator import AnswerEvaluator
from typing import List, Dict, Any
from fastapi.responses import JSONResponse
from vector_db import vectorize_and_store
from database import DataAlreadyExistsError  # 追加
from analysis import Analysis  # analysis.py をインポート
import uvicorn


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
        required_fields = ["temperature", "api_key", "task_name"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        temperature = data["temperature"]
        api_key = data["api_key"]
        task_name = data["task_name"]

        # QuestionGenerator を呼び出す
        question_generator = QuestionGenerator(
            temperature=temperature,
            api_key=api_key
        )

        # 質問と回答の生成
        results = question_generator.generate_questions_and_answers(task_name)

        # レスポンスの準備
        response_data = {
            "results": results,
            "total_tokens": question_generator.total_tokens,
            "total_processing_time": question_generator.total_processing_time
        }

        return JSONResponse(content=response_data, status_code=200)

    except DataAlreadyExistsError as e:
        # データが既に存在する場合のエラー処理
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        # データが見つからない場合のエラー処理
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as he:
        # 既に定義されたHTTPExceptionをそのまま返す
        raise he
    except Exception as e:
        logging.error(f"Error in /generate_questions: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)


@app.post("/evaluate_answers")
async def evaluate_answers(request: Request):
    try:
        # リクエストボディを取得
        data = await request.json()
    except Exception as e:
        logging.error(f"リクエストボディの解析中にエラーが発生しました: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

    try:
        required_fields = ["temperature", "api_key", "task_name"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        temperature = data["temperature"]
        api_key = data["api_key"]
        task_name = data["task_name"]

        # AnswerEvaluator をインスタンス化
        evaluator = AnswerEvaluator(api_key=api_key, temperature=temperature)

        # 回答の評価を実行
        results = evaluator.evaluate_answers(task_name)

        # 合計トークン数と処理時間を取得
        total_tokens = evaluator.total_tokens
        total_processing_time = evaluator.total_processing_time

        # レスポンスの準備
        response_data = {
            "results": results,
            "total_tokens": total_tokens,
            "total_processing_time": total_processing_time
        }

        return JSONResponse(content=response_data, status_code=200)

    except DataNotFoundError as e:
        # データが見つからない場合のエラー処理
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in /evaluate_answers: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)



@app.post("/vectorize_conversations")
async def vectorize_conversations(request: Request):
    try:
        data = await request.json()
        logging.info(f"Received data for vectorize_conversations: {data}")

        # 必要なフィールドが存在するかチェック
        required_fields = ["api_key", "task_name"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        api_key = data["api_key"]
        task_name = data["task_name"]

        # vector_db.py の関数を呼び出す
        result = vectorize_and_store(api_key=api_key, task_name=task_name)
        
        # エラー処理の追加
        if isinstance(result, dict) and result.get("status") == "error":
            logging.error(f"Vectorization error: {result.get('message')}")
            return JSONResponse(
                content={"detail": result.get("message")},
                status_code=400
            )

        # 成功時のレスポンス
        if isinstance(result, dict) and "count" in result:
            added_count = result["count"]
        else:
            added_count = result  # 後方互換性のため

        response_data = {
            "message": f"Successfully vectorized and stored {added_count} conversations."
        }

        return JSONResponse(content=response_data, status_code=200)

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in /vectorize_conversations: {str(e)}")
        # エラーメッセージをより詳細に
        error_message = f"ベクトル化処理中にエラーが発生しました: {str(e)}"
        return JSONResponse(
            content={"detail": error_message},
            status_code=500
        )


@app.get("/get_task_names")
async def get_task_names():
    try:
        analysis = Analysis(api_key="api_key不要")
        task_names = analysis.get_all_task_names()
        analysis.close()

        return JSONResponse(content={"task_names": task_names}, status_code=200)
    except Exception as e:
        logging.error(f"Error in /get_task_names: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)
    

@app.post("/get_cross_tab_data")
async def get_cross_tab_data(request: Request):
    try:
        data = await request.json()
        required_fields = ["api_key", "task_names"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")
        task_names = data.get("task_names", [])
        api_key = data["api_key"]

        if not task_names:
            raise HTTPException(status_code=400, detail="No task names specified.")

        analysis = Analysis(api_key=api_key)
        cross_tab_data = analysis.get_cross_tab_data(task_names)
        analysis.close()

        if cross_tab_data.empty:
            return JSONResponse(content={"data": []}, status_code=200)
        else:
            # Convert columns to string to ensure JSON serializability
            cross_tab_data.columns = cross_tab_data.columns.map(str)
            data = cross_tab_data.to_dict(orient='records')
            return JSONResponse(content={"data": data}, status_code=200)
    except Exception as e:
        logging.error(f"Error in /get_cross_tab_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching cross tab data: {e}")
    
    
@app.post("/perform_pca")
async def perform_pca(request: Request):
    try:
        data = await request.json()
        required_fields = ["api_key", "task_names"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")
        task_names = data.get("task_names", [])
        api_key = data["api_key"]

        if not task_names:
            raise HTTPException(status_code=400, detail="タスク名が指定されていません。")

        analysis = Analysis(api_key=api_key)
        pca_results = analysis.perform_pca(task_names)
        analysis.close()

        if not pca_results:
            return JSONResponse(content={"data": []}, status_code=200)
        else:
            return JSONResponse(content={"data": pca_results}, status_code=200)
    except Exception as e:
        logging.error(f"Error in /perform_pca: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PCA分析中にエラーが発生しました: {e}")


@app.post("/perform_svd_analysis")
async def perform_svd_analysis(request: Request):
    try:
        data = await request.json()
        required_fields = ["api_key", "task_names"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")
        task_names = data.get("task_names", [])
        api_key = data["api_key"]

        if not task_names:
            raise HTTPException(status_code=400, detail="タスク名が指定されていません。")

        analysis = Analysis(api_key=api_key)
        svd_results = analysis.perform_svd_analysis(task_names)
        analysis.close()

        if not svd_results:
            return JSONResponse(content={"data": []}, status_code=200)
        else:
            return JSONResponse(content={"data": svd_results}, status_code=200)
    except Exception as e:
        logging.error(f"Error in /perform_svd_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SVD分析中にエラーが発生しました: {e}")
    

@app.post("/get_evaluation_results")
async def get_evaluation_results(request: Request):
    try:
        data = await request.json()
        task_name = data.get("task_name", "")
        api_key = data.get("api_key", "")

        if not task_name:
            raise HTTPException(status_code=400, detail="タスク名が指定されていません。")

        analysis = Analysis(api_key=api_key)  # api_keyは不要なら固定文字列でもOK
        results = analysis.get_evaluation_results(task_name)
        analysis.close()

        return JSONResponse(content={"data": results}, status_code=200)
    except Exception as e:
        logging.error(f"Error in /get_evaluation_results: {str(e)}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)

    

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",              # "app.py"ファイル内の "app" インスタンスを指定
        host="0.0.0.0",         # ホスト指定 (必要に応じて変更)
        port=8000,              # ポート指定 (必要に応じて変更)
        timeout_keep_alive=99999,  # Keep-Alive接続のタイムアウト
        # 他にも必要なパラメータがあればここで指定可能
        workers=1
    )