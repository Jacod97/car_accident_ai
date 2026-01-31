"""
블랙박스 영상 분석 파이프라인

1차: 1fps로 전체 영상 분석 (매초 상황 + 사고 구간 감지)
2차: 사고 구간 ±1초를 5fps로 상세 분석
"""

from src.video_preprocessor import extract_frames_at_interval
from src.accident_detector import accident_detector_chain
from src.accident_detail_analyzer import accident_detail_chain, extract_accident_frames


video_path = r"C:\Project\toy\car_accident_ai\data\mp4\bb_1_190715_vehicle_149_235.mp4"


if __name__ == "__main__":
    if not video_path:
        print("video_path를 설정해주세요.")
        exit(1)

    print(f"영상 분석 시작: {video_path}\n")
    print("=" * 60)

    # 1차 분석 (1fps)
    print("[1차 분석] 1fps로 전체 영상 분석 중...")
    metadata, frames_1fps = extract_frames_at_interval(video_path)
    detect_result = accident_detector_chain.invoke(frames_1fps)

    accident_range = detect_result["accident_range"]
    descriptions = detect_result["descriptions"]

    # 사고 구간이 있으면 2차 분석
    detail_descriptions = {}
    detail_summary = ""

    if accident_range:
        print(f"[사고 감지] {accident_range[0]}초 ~ {accident_range[1]}초")
        print()
        print("[2차 분석] 사고 구간 5fps로 상세 분석 중...")

        frames_5fps, start_sec = extract_accident_frames(video_path, accident_range)
        detail_result = accident_detail_chain.invoke({
            "frames": frames_5fps,
            "start_sec": start_sec,
        })

        detail_descriptions = detail_result["details"]
        detail_summary = detail_result["summary"]

        # 사고 구간 ±1초는 1차 분석에서 제외
        detail_start = max(0, accident_range[0] - 1)
        detail_end = accident_range[1] + 1
        descriptions = {
            sec: desc for sec, desc in descriptions.items()
            if sec < detail_start or sec > detail_end
        }
    else:
        print("[사고 감지] 사고 구간 없음")

    # 결과 출력
    print()
    print("=" * 60)
    print("[분석 결과]")
    print("=" * 60)

    # 1차 + 2차 결과 합쳐서 시간순 정렬
    all_timestamps = []

    # 1차 분석 결과 (정수 초)
    for sec, desc in descriptions.items():
        all_timestamps.append((float(sec), f"{sec}초: {desc}"))

    # 2차 분석 결과 (소수점 초)
    for timestamp, desc in detail_descriptions.items():
        all_timestamps.append((timestamp, f"{timestamp:.1f}초: {desc}"))

    # 시간순 정렬 후 출력
    all_timestamps.sort(key=lambda x: x[0])

    for _, line in all_timestamps:
        print(line)

    # 사고 요약
    if detail_summary:
        print()
        print("=" * 60)
        print(f"[사고 요약] {detail_summary}")

    print("=" * 60)
    print("분석 완료")
