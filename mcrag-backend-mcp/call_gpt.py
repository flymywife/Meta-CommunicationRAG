# call_gpt.py

import openai
import json
import streamlit as st
import constants as c

class GPTClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    def call_gpt(self, messages: list, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
        """
        通常のGPT呼び出し用メソッド。

        Parameters:
        - messages (list): メッセージ履歴のリスト。
        - max_tokens (int): 最大トークン数。
        - temperature (float): 温度パラメータ。

        Returns:
        - dict: APIのレスポンス。
        """
        try:
            response = openai.ChatCompletion.create(
                model=c.MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=c.SEED
            )
            return response
        except Exception as e:
            st.error(f"ChatGPT呼び出し中にエラーが発生しました: {e}")
            return {}

    def call_gpt_function(self, messages: list, functions: list, function_call: dict, max_tokens: int = 2000, temperature: float = 0.7) -> dict:
        """
        Function Calling用のGPT呼び出しメソッド。

        Parameters:
        - messages (list): メッセージ履歴のリスト。
        - functions (list): Function Calling用の関数定義。
        - function_call (dict): Function Callingの設定。
        - max_tokens (int): 最大トークン数。
        - temperature (float): 温度パラメータ。

        Returns:
        - dict: APIのレスポンス。
        """
        try:
            response = openai.ChatCompletion.create(
                model=c.MODEL,
                messages=messages,
                functions=functions,
                function_call=function_call,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=c.SEED
            )
            return response
        except Exception as e:
            st.error(f"ChatGPT Function Calling中にエラーが発生しました: {e}")
            return {}
