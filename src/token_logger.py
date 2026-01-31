"""
토큰 사용량 로깅 모듈

Gemini countTokens API를 사용하여 실제 토큰 수 계산
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def count_tokens(content: list[dict], model_name: str = "gemini-2.5-flash") -> int:
    """
    Gemini countTokens API로 실제 토큰 수 계산

    Args:
        content: LangChain 메시지 content 형식
        model_name: 모델 이름

    Returns:
        토큰 수
    """
    model = genai.GenerativeModel(model_name)

    # LangChain content → Gemini 형식 변환
    gemini_parts = []
    for item in content:
        if item.get("type") == "text":
            gemini_parts.append(item["text"])
        elif item.get("type") == "image_url":
            # base64 이미지 추출
            url = item["image_url"]["url"]
            if url.startswith("data:image/jpeg;base64,"):
                import base64
                b64_data = url.replace("data:image/jpeg;base64,", "")
                image_bytes = base64.b64decode(b64_data)
                gemini_parts.append({
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                })

    result = model.count_tokens(gemini_parts)
    return result.total_tokens


def log_request_info(
    content: list[dict],
    frame_count: int,
    width: int,
    height: int,
) -> dict:
    """
    요청 정보 로깅 및 반환

    Args:
        content: LangChain 메시지 content
        frame_count: 프레임 수
        width: 프레임 너비
        height: 프레임 높이

    Returns:
        로깅 정보 dict
    """
    token_count = count_tokens(content)

    info = {
        "frame_count": frame_count,
        "resolution": f"{width}x{height}",
        "token_count": token_count,
    }

    print(f"[토큰 로그] 프레임: {frame_count}개, "
          f"해상도: {width}x{height}, "
          f"토큰: {token_count:,}")

    return info
