"""프레임을 LangChain 메시지 content 형식으로 변환"""

import base64

import cv2
import numpy as np

from src.config import JPEG_QUALITY, TARGET_WIDTH, TARGET_HEIGHT
from src.utils.logger import log_request_info


def encode_frame_to_base64(frame: np.ndarray, jpeg_quality: int = JPEG_QUALITY) -> str:
    """numpy 프레임을 base64 JPEG 문자열로 변환"""
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return base64.b64encode(encoded).decode("utf-8")


def build_content(
    prompt_text: str,
    frames: list[np.ndarray],
    timestamps: list[str],
) -> list[dict]:
    """
    프레임 + 타임스탬프 라벨을 LangChain 메시지 content로 변환

    Args:
        prompt_text: 프롬프트 텍스트
        frames: 프레임 리스트
        timestamps: 각 프레임에 대한 라벨 (예: "[프레임 1초]", "[3.0초]")

    Returns:
        LangChain HumanMessage content 형식
    """
    content = [{"type": "text", "text": prompt_text}]

    for frame, label in zip(frames, timestamps):
        b64 = encode_frame_to_base64(frame)
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    if frames:
        log_request_info(
            content=content,
            frame_count=len(frames),
            width=frames[0].shape[1],
            height=frames[0].shape[0],
        )

    return content
