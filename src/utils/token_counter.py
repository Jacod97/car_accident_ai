"""토큰 수 계산 (Gemini countTokens API)"""

from __future__ import annotations

import os
import base64

import google.generativeai as genai

from src.config import GEMINI_MODEL


class TokenCounter:
    """Gemini API 토큰 카운터.

    전역 상태(_configured 플래그)를 인스턴스로 캡슐화.
    생성자를 통해 모델명/API 키를 주입받음.
    """

    def __init__(self, model_name: str = GEMINI_MODEL, api_key: str | None = None):
        self._model_name = model_name
        self._api_key = api_key
        self._configured = False

    def _ensure_configured(self) -> None:
        if not self._configured:
            from dotenv import load_dotenv
            load_dotenv()
            key = self._api_key or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=key)
            self._configured = True

    def count_tokens(self, content: list[dict], model_name: str | None = None) -> int:
        """Gemini countTokens API로 실제 토큰 수 계산.

        Args:
            content: LangChain 메시지 content 형식.
            model_name: 모델명 오버라이드 (None이면 생성자 값 사용).
        """
        self._ensure_configured()

        model = genai.GenerativeModel(model_name or self._model_name)

        gemini_parts = []
        for item in content:
            if item.get("type") == "text":
                gemini_parts.append(item["text"])
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:image/jpeg;base64,"):
                    b64_data = url.replace("data:image/jpeg;base64,", "")
                    image_bytes = base64.b64decode(b64_data)
                    gemini_parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes,
                    })

        result = model.count_tokens(gemini_parts)
        return result.total_tokens


# Backward-compatible
_default_counter = TokenCounter()


def count_tokens(content: list[dict], model_name: str = GEMINI_MODEL) -> int:
    return _default_counter.count_tokens(content, model_name)
