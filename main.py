"""
블랙박스 영상 분석 파이프라인

1차: 1fps로 전체 영상 분석 (매초 상황 + 사고 구간 감지)
2차: 사고 구간 ±1초를 상세 분석
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

from src.video.preprocessor import FrameExtractor
from src.analysis.detector import AccidentDetector
from src.analysis.detail_analyzer import AccidentDetailAnalyzer
from src.models import DetectorResult, DetailResult
from src.config import DETECTOR_INTERVAL_SEC

logging.basicConfig(level=logging.INFO, format="%(message)s")


class AccidentPipeline:
    """2단계 사고 분석 파이프라인 오케스트레이터.

    의존성 주입: detector, detail_analyzer, extractor를 생성자로 주입.
    None이면 기본 인스턴스를 생성.
    """

    def __init__(
        self,
        detector: AccidentDetector | None = None,
        detail_analyzer: AccidentDetailAnalyzer | None = None,
        extractor: FrameExtractor | None = None,
    ):
        self._detector = detector or AccidentDetector()
        self._detail_analyzer = detail_analyzer or AccidentDetailAnalyzer()
        self._extractor = extractor or FrameExtractor()

    def run(self, video_path: str | Path) -> dict:
        """전체 파이프라인 실행. 1차 + 2차(조건부) 분석 수행 후 결과 반환."""
        print(f"영상 분석 시작: {video_path}\n")
        print("=" * 60)

        # 1차 분석
        detect_result = self._run_detection(video_path)
        descriptions = dict(detect_result.descriptions)
        accident_range = detect_result.accident_range

        # 2차 분석 (사고 감지 시)
        detail_result = None

        if accident_range:
            detail_result = self._run_detail(video_path, accident_range)

            # 사고 구간 ±1초는 1차 분석에서 제외 (2차 결과로 대체)
            detail_start = max(0, accident_range[0] - 1)
            detail_end = accident_range[1] + 1
            descriptions = {
                sec: desc for sec, desc in descriptions.items()
                if sec < detail_start or sec > detail_end
            }
        else:
            print("[사고 감지] 사고 구간 없음")

        # 결과 출력
        self._print_results(descriptions, detail_result)

        return {
            "detector": detect_result,
            "detail": detail_result,
        }

    def _run_detection(self, video_path: str | Path) -> DetectorResult:
        """1차 분석: 1fps 간격으로 전체 영상 분석"""
        print("[1차 분석] 1fps로 전체 영상 분석 중...")
        metadata, frames = self._extractor.extract(
            video_path, interval_sec=DETECTOR_INTERVAL_SEC
        )
        return self._detector.analyze(frames)

    def _run_detail(self, video_path: str | Path, accident_range: list[int]) -> DetailResult:
        """2차 분석: 사고 구간 ± 패딩을 고FPS로 상세 분석"""
        print(f"[사고 감지] {accident_range[0]}초 ~ {accident_range[1]}초")
        print("\n[2차 분석] 사고 구간 상세 분석 중...")
        frames, start_sec = self._extractor.extract(
            video_path, accident_range=accident_range
        )
        return self._detail_analyzer.analyze({
            "frames": frames,
            "start_sec": start_sec,
        })

    def _print_results(self, descriptions: dict[int, str], detail_result: DetailResult | None) -> None:
        """분석 결과 포맷팅 및 출력"""
        print()
        print("=" * 60)
        print("[분석 결과]")
        print("=" * 60)

        # 1차 + 2차 결과 시간순 정렬
        all_timestamps = []

        for sec, desc in descriptions.items():
            all_timestamps.append((float(sec), f"{sec}초: {desc}"))

        if detail_result:
            for timestamp, desc in detail_result.details.items():
                all_timestamps.append((timestamp, f"{timestamp:.1f}초: {desc}"))

        all_timestamps.sort(key=lambda x: x[0])

        for _, line in all_timestamps:
            print(line)

        if detail_result and detail_result.summary:
            print()
            print("=" * 60)
            print(f"[사고 요약] {detail_result.summary}")

        print("=" * 60)
        print("분석 완료")


def run_pipeline(video_path: str) -> None:
    """Backward-compatible 함수형 진입점."""
    pipeline = AccidentPipeline()
    pipeline.run(video_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python main.py <영상 경로>")
        sys.exit(1)

    run_pipeline(sys.argv[1])
