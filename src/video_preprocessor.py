import cv2
import numpy as np
from pathlib import Path

from src.video_loader import VideoMetadata, VideoLoader
from src.video_validator import check_duration, check_resolution, check_codec, check_fps
from src.config import TARGET_WIDTH, TARGET_HEIGHT, MAX_VIDEO_SEC, DETECTOR_INTERVAL_SEC


def validate_or_raise(metadata: VideoMetadata) -> None:
    """전처리 전 검증. 처리 불가능한 영상은 에러 발생"""
    duration_status = check_duration(metadata.duration_seconds)
    if duration_status == -1:
        raise ValueError(f"영상이 너무 짧습니다: {metadata.duration_seconds}초")

    if not check_codec(metadata.codec):
        raise ValueError(f"지원하지 않는 코덱입니다: {metadata.codec}")

    if not check_fps(metadata.fps):
        raise ValueError(f"FPS가 너무 낮습니다: {metadata.fps}")


def calculate_trim_range(total_frames: int, fps: float, max_sec: float = 20.0) -> tuple[int, int]:
    """
    영상이 길 경우 자를 프레임 범위 계산 (뒷부분 유지)

    Returns:
        (시작 프레임, 끝 프레임)
    """
    duration = total_frames / fps

    if duration <= max_sec:
        return 0, total_frames

    # 뒷부분만 유지: 예) 30초 영상 -> 10~30초 (프레임 기준)
    keep_frames = int(max_sec * fps)
    start_frame = total_frames - keep_frames

    return start_frame, total_frames


def resize_frame(frame: np.ndarray, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> np.ndarray:
    """프레임을 목표 해상도로 리사이즈"""
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def extract_frames(
    video_path: str | Path,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    max_sec: float = MAX_VIDEO_SEC,
) -> tuple[VideoMetadata, list[np.ndarray]]:
    """
    영상에서 프레임 추출 및 전처리

    Args:
        video_path: 영상 파일 경로
        target_width: 목표 너비
        target_height: 목표 높이
        max_sec: 최대 영상 길이 (초과시 뒷부분만 유지)

    Returns:
        (원본 메타데이터, 전처리된 프레임 리스트)

    Raises:
        ValueError: 영상이 짧거나, 코덱 미지원, FPS 부족시
    """
    loader = VideoLoader(video_path)
    loader.load()
    metadata = loader.get_metadata()

    # 검증 (실패시 에러)
    validate_or_raise(metadata)

    # 자를 범위 계산
    start_frame, end_frame = calculate_trim_range(
        metadata.total_frames,
        metadata.fps,
        max_sec
    )

    # 리사이즈 필요 여부
    need_resize = not check_resolution(metadata.width, metadata.height, target_width, target_height)

    # 프레임 추출
    frames = []
    cap = loader._cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if need_resize:
            frame = resize_frame(frame, target_width, target_height)

        frames.append(frame)

    cap.release()
    return metadata, frames


def extract_frames_at_interval(
    video_path: str | Path,
    interval_sec: float = DETECTOR_INTERVAL_SEC,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    max_sec: float = MAX_VIDEO_SEC,
) -> tuple[VideoMetadata, list[np.ndarray]]:
    """
    일정 간격으로 프레임 추출 (LLM 입력용)

    Args:
        video_path: 영상 파일 경로
        interval_sec: 추출 간격 (초). 기본 1초
        target_width: 목표 너비
        target_height: 목표 높이
        max_sec: 최대 영상 길이

    Returns:
        (원본 메타데이터, 추출된 프레임 리스트)
    """
    loader = VideoLoader(video_path)
    loader.load()
    metadata = loader.get_metadata()

    # 검증
    validate_or_raise(metadata)

    # 자를 범위 계산
    start_frame, end_frame = calculate_trim_range(
        metadata.total_frames,
        metadata.fps,
        max_sec
    )

    # 리사이즈 필요 여부
    need_resize = not check_resolution(metadata.width, metadata.height, target_width, target_height)

    # 간격에 맞는 프레임 인덱스 계산
    frame_interval = int(metadata.fps * interval_sec)

    frames = []
    cap = loader._cap

    current_frame = start_frame
    while current_frame < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        if need_resize:
            frame = resize_frame(frame, target_width, target_height)

        frames.append(frame)
        current_frame += frame_interval

    cap.release()
    return metadata, frames


if __name__ == "__main__":
    from pathlib import Path
    from src.video_loader import load_videos_from_directory

    data_dir = Path(__file__).parent.parent / "data" / "mp4"
    videos = load_videos_from_directory(data_dir)

    print(f"총 {len(videos)}개 영상 전처리 테스트\n")

    for video in videos[:3]:
        try:
            metadata, frames = extract_frames_at_interval(
                video.video_path,
                interval_sec=1.0,
            )
            print(f"✓ {metadata.file_name}")
            print(f"  원본: {metadata.width}x{metadata.height}, {metadata.duration_seconds}초")
            print(f"  추출된 프레임: {len(frames)}개")
            if frames:
                print(f"  프레임 크기: {frames[0].shape[1]}x{frames[0].shape[0]}")
            print()
        except ValueError as e:
            print(f"✗ {video.video_path.name}: {e}\n")
