"""AI 분석 + 파이프라인 실행

사용법: python legacy/analyze.py <영상 경로>
"""

import sys
import base64
import cv2
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from video import extract_frames_1fps, extract_accident_frames


load_dotenv()


# ============ 설정 ============

GEMINI_MODEL = "gemini-2.5-flash"
JPEG_QUALITY = 70
DETECTOR_INTERVAL_SEC = 1.0
DETAIL_FPS = 3
DETAIL_INTERVAL = round(1.0 / DETAIL_FPS, 2)

model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)


# ============ 프롬프트 ============

DETECTOR_PROMPT = f"""당신은 블랙박스 영상 분석 전문가입니다.
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

DETAIL_PROMPT = f"""당신은 블랙박스 영상 분석 전문가입니다.
아래 프레임들은 사고 구간을 {DETAIL_INTERVAL}초 간격으로 추출한 이미지입니다.

## 임무
각 프레임마다 상세한 상황을 분석하세요.
- 차량 위치, 속도 변화, 방향
- 충돌 직전/직후 상황
- 과실 판단에 도움이 되는 정보

## 응답 형식 (반드시 이 형식으로)
[상세 분석]
0.0초: (상세 설명)
0.2초: (상세 설명)
0.4초: (상세 설명)
...

[사고 요약]
(사고 경위 요약)

## 프레임 이미지들:
"""


# ============ content 빌드 ============

def frames_to_base64(frame):
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(encoded).decode("utf-8")


def build_detector_content(frames):
    content = [{"type": "text", "text": DETECTOR_PROMPT}]
    for i, frame in enumerate(frames):
        sec = int((i + 1) * DETECTOR_INTERVAL_SEC)
        b64 = frames_to_base64(frame)
        content.append({"type": "text", "text": f"[프레임 {sec}초]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def build_detail_content(data):
    frames = data["frames"]
    start_sec = data["start_sec"]
    content = [{"type": "text", "text": DETAIL_PROMPT}]
    for i, frame in enumerate(frames):
        timestamp = start_sec + (i * DETAIL_INTERVAL)
        b64 = frames_to_base64(frame)
        content.append({"type": "text", "text": f"[{timestamp:.1f}초]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


# ============ 응답 파싱 ============

def parse_detector_response(text):
    descriptions = {}
    accident_range = None
    in_description = False
    in_accident = False
    start = None
    end = None

    for line in text.strip().split("\n"):
        line = line.strip()

        if "[상황 설명]" in line:
            in_description, in_accident = True, False
            continue
        if "[사고 구간]" in line:
            in_description, in_accident = False, True
            continue

        if in_description and "초:" in line:
            parts = line.split("초:", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                descriptions[int(parts[0].strip())] = parts[1].strip()

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
        accident_range = [start, end]

    return descriptions, accident_range, text


def parse_detail_response(text):
    details = {}
    summary_lines = []
    in_detail = False
    in_summary = False

    for line in text.strip().split("\n"):
        line = line.strip()

        if "[상세 분석]" in line:
            in_detail, in_summary = True, False
            continue
        if "[사고 요약]" in line:
            in_detail, in_summary = False, True
            continue

        if in_detail and "초:" in line:
            parts = line.split("초:", 1)
            if len(parts) == 2:
                try:
                    details[float(parts[0].strip())] = parts[1].strip()
                except ValueError:
                    pass

        if in_summary and line:
            summary_lines.append(line)

    return details, " ".join(summary_lines), text


# ============ 체인 ============

detector_chain = (
    RunnableLambda(build_detector_content)
    | RunnableLambda(lambda content: [HumanMessage(content=content)])
    | model
    | StrOutputParser()
    | RunnableLambda(lambda text: parse_detector_response(text))
)

detail_chain = (
    RunnableLambda(build_detail_content)
    | RunnableLambda(lambda content: [HumanMessage(content=content)])
    | model
    | StrOutputParser()
    | RunnableLambda(lambda text: parse_detail_response(text))
)


# ============ 메인 실행 ============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python legacy/analyze.py <영상 경로>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"영상 분석 시작: {video_path}\n")
    print("=" * 60)

    # 1차 분석
    print("[1차 분석] 1fps로 전체 영상 분석 중...")
    metadata, frames_1fps = extract_frames_1fps(video_path)
    descriptions, accident_range, raw1 = detector_chain.invoke(frames_1fps)

    # 2차 분석 (사고 감지 시)
    detail_descriptions = {}
    detail_summary = ""

    if accident_range:
        print(f"[사고 감지] {accident_range[0]}초 ~ {accident_range[1]}초\n")
        print("[2차 분석] 사고 구간 상세 분석 중...")

        detail_frames, start_sec = extract_accident_frames(video_path, accident_range)
        detail_descriptions, detail_summary, raw2 = detail_chain.invoke({
            "frames": detail_frames,
            "start_sec": start_sec,
        })

        # 사고 구간 ±1초는 1차에서 제외
        detail_start = max(0, accident_range[0] - 1)
        detail_end = accident_range[1] + 1
        descriptions = {
            sec: desc for sec, desc in descriptions.items()
            if sec < detail_start or sec > detail_end
        }
    else:
        print("[사고 감지] 사고 구간 없음")

    # 결과 출력
    print("\n" + "=" * 60)
    print("[분석 결과]")
    print("=" * 60)

    all_timestamps = []
    for sec, desc in descriptions.items():
        all_timestamps.append((float(sec), f"{sec}초: {desc}"))
    for ts, desc in detail_descriptions.items():
        all_timestamps.append((ts, f"{ts:.1f}초: {desc}"))

    all_timestamps.sort(key=lambda x: x[0])
    for _, line in all_timestamps:
        print(line)

    if detail_summary:
        print("\n" + "=" * 60)
        print(f"[사고 요약] {detail_summary}")

    print("=" * 60)
    print("분석 완료")
