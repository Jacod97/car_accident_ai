"""영상 로드, 검증, 프레임 추출"""

import cv2
import numpy as np
from pathlib import Path


# ============ 설정 ============

TARGET_WIDTH = 512
TARGET_HEIGHT = 288
MAX_VIDEO_SEC = 20.0
MIN_VIDEO_SEC = 10.0
MIN_FPS = 10.0
DETECTOR_INTERVAL_SEC = 1.0
DETAIL_FPS = 3
ACCIDENT_PADDING_SEC = 1

SUPPORTED_CODECS = {
    "avc1", "h264", "H264",
    "hevc", "hev1", "H265",
    "mp4v", "MP4V",
    "XVID", "xvid",
    "DIVX", "divx",
    "MJPG", "mjpg",
    "VP80", "VP90",
    "WMV1", "WMV2", "WMV3",
    "FMP4",
}


# ============ 영상 로드 + 메타데이터 ============

def load_video(video_path):
    """영상을 열고 (cap, metadata dict) 반환"""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    duration = total_frames / fps if fps > 0 else 0

    metadata = {
        "file_name": path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration": round(duration, 2),
        "codec": codec,
    }

    return cap, metadata


# ============ 검증 ============

def validate(metadata):
    """영상 검증. 실패 시 ValueError"""
    if metadata["duration"] < MIN_VIDEO_SEC:
        raise ValueError(f"영상이 너무 짧습니다: {metadata['duration']}초")
    if metadata["codec"].strip() not in SUPPORTED_CODECS:
        raise ValueError(f"지원하지 않는 코덱: {metadata['codec']}")
    if metadata["fps"] <= MIN_FPS:
        raise ValueError(f"FPS가 너무 낮습니다: {metadata['fps']}")


# ============ 프레임 추출 ============

def extract_frames_1fps(video_path):
    """1fps 간격으로 프레임 추출. (metadata, frames) 반환"""
    cap, metadata = load_video(video_path)
    validate(metadata)

    fps = metadata["fps"]
    total = metadata["total_frames"]

    # 20초 초과 시 뒷부분만 유지
    duration = total / fps
    if duration > MAX_VIDEO_SEC:
        start = total - int(MAX_VIDEO_SEC * fps)
    else:
        start = 0

    need_resize = (metadata["width"] != TARGET_WIDTH or metadata["height"] != TARGET_HEIGHT)
    frame_interval = int(fps * DETECTOR_INTERVAL_SEC)

    frames = []
    current = start
    while current < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret:
            break
        if need_resize:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        current += frame_interval

    cap.release()
    return metadata, frames


def extract_accident_frames(video_path, accident_range):
    """사고 구간 ± 1초를 3fps로 추출. (frames, start_sec) 반환"""
    cap, metadata = load_video(video_path)
    fps = metadata["fps"]

    start_sec = max(0, accident_range[0] - ACCIDENT_PADDING_SEC)
    end_sec = min(metadata["duration"], accident_range[1] + ACCIDENT_PADDING_SEC)

    frame_interval = fps / DETAIL_FPS
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    need_resize = (metadata["width"] != TARGET_WIDTH or metadata["height"] != TARGET_HEIGHT)

    frames = []
    current = start_frame
    while current < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current))
        ret, frame = cap.read()
        if not ret:
            break
        if need_resize:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        current += frame_interval

    cap.release()
    return frames, start_sec
