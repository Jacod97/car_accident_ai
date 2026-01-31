"""
사고 상세 분석 AI (2차 분석)

사고 구간 ± 1초를 상세 분석합니다.
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

from src.video_loader import VideoLoader
from src.config import JPEG_QUALITY, TARGET_WIDTH, TARGET_HEIGHT, DETAIL_FPS
from src.token_logger import log_request_info


load_dotenv()


# ============ 모델 ============

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 프레임 간격 계산
DETAIL_INTERVAL = round(1.0 / DETAIL_FPS, 2)


# ============ 프롬프트 ============

PROMPT = f"""당신은 블랙박스 영상 분석 전문가입니다.
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


# ============ 체인 구성 ============

frame_to_base64 = RunnableLambda(
    lambda frame: base64.b64encode(
        cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
    ).decode("utf-8")
)

# (frames, start_sec) 튜플을 받아서 content로 변환
frames_to_content = RunnableLambda(
    lambda data: _build_content(data["frames"], data["start_sec"])
)


def _build_content(frames: list[np.ndarray], start_sec: float) -> list[dict]:
    """프레임과 시작 시간을 받아 메시지 content 생성 + 로깅"""
    content = [{"type": "text", "text": PROMPT}]

    for i, frame in enumerate(frames):
        timestamp = start_sec + (i * DETAIL_INTERVAL)
        b64 = base64.b64encode(
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
        ).decode("utf-8")

        content.append({"type": "text", "text": f"[{timestamp:.1f}초]"})
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


content_to_message = RunnableLambda(
    lambda content: [HumanMessage(content=content)]
)


def _parse_detail_response(text: str) -> dict:
    """응답 파싱"""
    result = {
        "details": {},
        "summary": "",
        "raw_response": text,
    }

    lines = text.strip().split("\n")
    in_detail = False
    in_summary = False
    summary_lines = []

    for line in lines:
        line = line.strip()

        if "[상세 분석]" in line:
            in_detail = True
            in_summary = False
            continue

        if "[사고 요약]" in line:
            in_detail = False
            in_summary = True
            continue

        if in_detail and "초:" in line:
            # "0.2초: 설명" 형태 파싱
            parts = line.split("초:", 1)
            if len(parts) == 2:
                try:
                    timestamp = float(parts[0].strip())
                    result["details"][timestamp] = parts[1].strip()
                except ValueError:
                    pass

        if in_summary and line:
            summary_lines.append(line)

    result["summary"] = " ".join(summary_lines)
    return result


parse_response = RunnableLambda(
    lambda text: _parse_detail_response(text)
)


# ============ 메인 체인 ============

accident_detail_chain = (
    frames_to_content
    | content_to_message
    | model
    | StrOutputParser()
    | parse_response
)


# ============ 프레임 추출 함수 ============

def extract_accident_frames(
    video_path: str | Path,
    accident_range: list[int],
    fps: int = DETAIL_FPS,
    padding_sec: int = 1,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> tuple[list[np.ndarray], float]:
    """
    사고 구간 ± padding_sec 만큼의 프레임을 fps로 추출

    Args:
        video_path: 영상 경로
        accident_range: [시작초, 종료초]
        fps: 추출할 fps (기본 5)
        padding_sec: 앞뒤로 추가할 초 (기본 1)

    Returns:
        (프레임 리스트, 시작 시간(초))
    """
    start_sec = max(0, accident_range[0] - padding_sec)
    end_sec = accident_range[1] + padding_sec

    loader = VideoLoader(video_path)
    loader.load()
    metadata = loader.get_metadata()
    cap = loader._cap

    # 실제 종료 시간은 영상 길이를 넘지 않도록
    end_sec = min(end_sec, metadata.duration_seconds)

    # 프레임 간격 계산
    frame_interval = metadata.fps / fps
    start_frame = int(start_sec * metadata.fps)
    end_frame = int(end_sec * metadata.fps)

    frames = []
    current_frame = start_frame

    while current_frame < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame))
        ret, frame = cap.read()

        if not ret:
            break

        # 리사이즈
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))

        frames.append(frame)
        current_frame += frame_interval

    cap.release()
    return frames, start_sec


if __name__ == "__main__":
    from src.video_loader import load_videos_from_directory
    from src.video_preprocessor import extract_frames_at_interval
    from src.accident_detector import accident_detector_chain

    data_dir = Path(__file__).parent.parent / "data" / "mp4"
    videos = load_videos_from_directory(data_dir)

    print("사고 상세 분석 테스트\n")

    if videos:
        video = videos[0]
        print(f"1차 분석 중: {video.video_path.name}")

        try:
            # 1차 분석
            metadata, frames_1fps = extract_frames_at_interval(video.video_path)
            detect_result = accident_detector_chain.invoke(frames_1fps)

            if detect_result["accident_range"]:
                print(f"사고 구간 감지: {detect_result['accident_range']}")

                # 2차 상세 분석
                print(f"\n2차 상세 분석 중...")
                frames_5fps, start_sec = extract_accident_frames(
                    video.video_path,
                    detect_result["accident_range"],
                )

                detail_result = accident_detail_chain.invoke({
                    "frames": frames_5fps,
                    "start_sec": start_sec,
                })

                print(f"\n[상세 분석]")
                for timestamp, desc in sorted(detail_result["details"].items()):
                    print(f"  {timestamp:.1f}초: {desc}")

                print(f"\n[사고 요약]: {detail_result['summary']}")
            else:
                print("사고 구간이 감지되지 않았습니다.")

        except Exception as e:
            print(f"에러: {e}")
