"""프레임 추출 및 전처리"""

from __future__ import annotations

from typing import overload
from pathlib import Path

import cv2
import numpy as np

from src.models import VideoMetadata
from src.video.loader import VideoLoader
from src.video.validator import validate_or_raise, check_resolution
from src.config import (
    TARGET_WIDTH, TARGET_HEIGHT, MAX_VIDEO_SEC,
    DETECTOR_INTERVAL_SEC, DETAIL_FPS, ACCIDENT_PADDING_SEC,
)


# ---- 순수 함수 (상태 없음, 모듈 레벨 유지) ----

def calculate_trim_range(total_frames: int, fps: float, max_sec: float = MAX_VIDEO_SEC) -> tuple[int, int]:
    """영상이 길 경우 자를 프레임 범위 계산 (뒷부분 유지)"""
    duration = total_frames / fps
    if duration <= max_sec:
        return 0, total_frames
    keep_frames = int(max_sec * fps)
    start_frame = total_frames - keep_frames
    return start_frame, total_frames


def resize_frame(frame: np.ndarray, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> np.ndarray:
    """프레임을 목표 해상도로 리사이즈"""
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


# ---- FrameExtractor 클래스 ----

class FrameExtractor:
    """통합 프레임 추출기.

    @overload를 사용한 메서드 오버로딩으로 3가지 추출 모드를 하나의 extract() 메서드로 제공:
    1. extract(video_path)                    → 전체 프레임
    2. extract(video_path, interval_sec=1.0)  → 간격 기반 추출
    3. extract(video_path, accident_range=[6,7]) → 사고 구간 추출
    """

    def __init__(
        self,
        target_width: int = TARGET_WIDTH,
        target_height: int = TARGET_HEIGHT,
        max_sec: float = MAX_VIDEO_SEC,
        detail_fps: int = DETAIL_FPS,
        padding_sec: int = ACCIDENT_PADDING_SEC,
    ):
        self._target_width = target_width
        self._target_height = target_height
        self._max_sec = max_sec
        self._detail_fps = detail_fps
        self._padding_sec = padding_sec

    # ---- @overload 시그니처 (타입 체커용) ----

    @overload
    def extract(
        self, video_path: str | Path,
    ) -> tuple[VideoMetadata, list[np.ndarray]]: ...

    @overload
    def extract(
        self, video_path: str | Path,
        *, interval_sec: float,
    ) -> tuple[VideoMetadata, list[np.ndarray]]: ...

    @overload
    def extract(
        self, video_path: str | Path,
        *, accident_range: list[int],
    ) -> tuple[list[np.ndarray], float]: ...

    # ---- 실제 구현 (디스패처) ----

    def extract(
        self,
        video_path: str | Path,
        *,
        interval_sec: float | None = None,
        accident_range: list[int] | None = None,
    ) -> tuple[VideoMetadata, list[np.ndarray]] | tuple[list[np.ndarray], float]:
        """오버로딩된 프레임 추출.

        - 인자 없음: 전체 프레임 추출
        - interval_sec: 일정 간격 추출
        - accident_range: 사고 구간 ± 패딩 추출
        """
        if accident_range is not None:
            return self._extract_accident(video_path, accident_range)
        elif interval_sec is not None:
            return self._extract_at_interval(video_path, interval_sec)
        else:
            return self._extract_all(video_path)

    # ---- 공유 private 메서드 ----

    def _load_and_validate(self, video_path: str | Path) -> tuple[VideoLoader, VideoMetadata]:
        """모든 추출 모드의 공통 첫 단계: 로드 + 검증"""
        loader = VideoLoader(video_path)
        loader.load()
        metadata = loader.get_metadata()
        validate_or_raise(metadata)
        return loader, metadata

    def _needs_resize(self, width: int, height: int) -> bool:
        return not check_resolution(width, height, self._target_width, self._target_height)

    def _maybe_resize(self, frame: np.ndarray, need_resize: bool) -> np.ndarray:
        if need_resize:
            return resize_frame(frame, self._target_width, self._target_height)
        return frame

    # ---- 추출 전략 ----

    def _extract_all(self, video_path: str | Path) -> tuple[VideoMetadata, list[np.ndarray]]:
        """전체 프레임 추출 (트림 후)"""
        loader, metadata = self._load_and_validate(video_path)
        start_frame, end_frame = calculate_trim_range(metadata.total_frames, metadata.fps, self._max_sec)
        need_resize = self._needs_resize(metadata.width, metadata.height)

        frames = []
        cap = loader._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self._maybe_resize(frame, need_resize))

        cap.release()
        return metadata, frames

    def _extract_at_interval(self, video_path: str | Path, interval_sec: float) -> tuple[VideoMetadata, list[np.ndarray]]:
        """일정 간격으로 프레임 추출"""
        loader, metadata = self._load_and_validate(video_path)
        start_frame, end_frame = calculate_trim_range(metadata.total_frames, metadata.fps, self._max_sec)
        need_resize = self._needs_resize(metadata.width, metadata.height)
        frame_interval = int(metadata.fps * interval_sec)

        frames = []
        cap = loader._cap
        current_frame = start_frame

        while current_frame < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self._maybe_resize(frame, need_resize))
            current_frame += frame_interval

        cap.release()
        return metadata, frames

    def _extract_accident(self, video_path: str | Path, accident_range: list[int]) -> tuple[list[np.ndarray], float]:
        """사고 구간 ± 패딩 프레임 추출"""
        start_sec = max(0, accident_range[0] - self._padding_sec)
        end_sec = accident_range[1] + self._padding_sec

        loader = VideoLoader(video_path)
        loader.load()
        metadata = loader.get_metadata()
        cap = loader._cap

        end_sec = min(end_sec, metadata.duration_seconds)

        frame_interval = metadata.fps / self._detail_fps
        start_frame = int(start_sec * metadata.fps)
        end_frame = int(end_sec * metadata.fps)
        need_resize = self._needs_resize(metadata.width, metadata.height)

        frames = []
        current_frame = start_frame

        while current_frame < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame))
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self._maybe_resize(frame, need_resize))
            current_frame += frame_interval

        cap.release()
        return frames, start_sec


# ---- Backward-compatible 모듈 레벨 함수 ----

_default_extractor = FrameExtractor()


def extract_frames(
    video_path: str | Path,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    max_sec: float = MAX_VIDEO_SEC,
) -> tuple[VideoMetadata, list[np.ndarray]]:
    return _default_extractor.extract(video_path)


def extract_frames_at_interval(
    video_path: str | Path,
    interval_sec: float = DETECTOR_INTERVAL_SEC,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    max_sec: float = MAX_VIDEO_SEC,
) -> tuple[VideoMetadata, list[np.ndarray]]:
    return _default_extractor.extract(video_path, interval_sec=interval_sec)


def extract_accident_frames(
    video_path: str | Path,
    accident_range: list[int],
    fps: int = DETAIL_FPS,
    padding_sec: int = ACCIDENT_PADDING_SEC,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> tuple[list[np.ndarray], float]:
    return _default_extractor.extract(video_path, accident_range=accident_range)
