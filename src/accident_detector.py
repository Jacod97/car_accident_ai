"""
사고 감지 AI (1차 분석)

프레임 간격으로 영상을 분석하여:
- 매 프레임마다 상황 설명
- 사고 구간 반환
"""

import base64
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.config import JPEG_QUALITY, TARGET_WIDTH, TARGET_HEIGHT, DETECTOR_INTERVAL_SEC
from src.token_logger import log_request_info


load_dotenv()


# ============ 모델 ============

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ============ 프롬프트 ============

PROMPT = f"""당신은 블랙박스 영상 분석 전문가입니다.
아래 프레임들은 블랙박스 영상에서 {DETECTOR_INTERVAL_SEC}초 간격으로 추출한 이미지입니다.

## 임무
1. 각 프레임(초)마다 상황을 설명하세요
2. 사고(충돌) 시점을 찾으세요

## 사고 구간 판단 기준
- 시작: 충돌 직전 마지막 프레임 (아직 충돌 안 됨)
- 종료: 충돌 직후 첫 프레임 (충돌 발생 상태)
- 시작과 종료는 반드시 연속된 프레임이어야 함 (예: 6초, 7초)
- 사고가 없으면 "없음"

## 응답 형식 (반드시 이 형식으로)
[상황 설명]
1초: (상황 설명)
2초: (상황 설명)
...

[사고 구간]
시작: (숫자, 사고 없으면 "없음")
종료: (숫자, 사고 없으면 "없음")

## 프레임 이미지들:
"""


# ============ 체인 구성 ============

def _build_content(frames: list[np.ndarray]) -> list[dict]:
    """프레임 리스트를 메시지 content로 변환 + 로깅"""
    content = [{"type": "text", "text": PROMPT}]

    for i, frame in enumerate(frames):
        b64 = base64.b64encode(
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
        ).decode("utf-8")

        sec = int((i + 1) * DETECTOR_INTERVAL_SEC)
        content.append({"type": "text", "text": f"[프레임 {sec}초]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # 로깅
    if frames:
        log_request_info(
            content=content,
            frame_count=len(frames),
            width=frames[0].shape[1],
            height=frames[0].shape[0],
        )

    return content


frames_to_content = RunnableLambda(_build_content)

content_to_message = RunnableLambda(
    lambda content: [HumanMessage(content=content)]
)

parse_response = RunnableLambda(
    lambda text: _parse_detector_response(text)
)


def _parse_detector_response(text: str) -> dict:
    """응답 파싱"""
    result = {
        "descriptions": {},
        "accident_range": None,
        "raw_response": text,
    }

    lines = text.strip().split("\n")
    in_description = False
    in_accident = False
    start = None
    end = None

    for line in lines:
        line = line.strip()

        if "[상황 설명]" in line:
            in_description = True
            in_accident = False
            continue

        if "[사고 구간]" in line:
            in_description = False
            in_accident = True
            continue

        if in_description and "초:" in line:
            # "1초: 설명" 형태 파싱
            parts = line.split("초:", 1)
            if len(parts) == 2:
                sec = parts[0].strip()
                if sec.isdigit():
                    result["descriptions"][int(sec)] = parts[1].strip()

        if in_accident:
            if line.startswith("시작:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    start = int(val)
            elif line.startswith("종료:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    end = int(val)

    if start is not None and end is not None:
        result["accident_range"] = [start, end]

    return result


# ============ 메인 체인 ============

accident_detector_chain = (
    frames_to_content
    | content_to_message
    | model
    | StrOutputParser()
    | parse_response
)


if __name__ == "__main__":
    from src.video_loader import load_videos_from_directory
    from src.video_preprocessor import extract_frames_at_interval

    data_dir = Path(__file__).parent.parent / "data" / "mp4"
    videos = load_videos_from_directory(data_dir)

    print("사고 감지 AI 테스트\n")

    if videos:
        video = videos[0]
        print(f"분석 중: {video.video_path.name}")

        try:
            metadata, frames = extract_frames_at_interval(video.video_path)
            result = accident_detector_chain.invoke(frames)

            print(f"\n[상황 설명]")
            for sec, desc in sorted(result["descriptions"].items()):
                print(f"  {sec}초: {desc}")

            print(f"\n[사고 구간]: {result['accident_range']}")
        except Exception as e:
            print(f"에러: {e}")
