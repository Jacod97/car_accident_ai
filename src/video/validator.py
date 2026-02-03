from src.config import TARGET_WIDTH, TARGET_HEIGHT, MAX_VIDEO_SEC, MIN_VIDEO_SEC, MIN_FPS
from src.models import VideoMetadata


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


def check_duration(duration_seconds: float, min_sec: float = MIN_VIDEO_SEC, max_sec: float = MAX_VIDEO_SEC) -> int:
    """영상 길이 검사. -1: 짧음, 0: 적절, 1: 김"""
    if duration_seconds < min_sec:
        return -1
    elif duration_seconds > max_sec:
        return 1
    else:
        return 0


def check_resolution(width: int, height: int, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> bool:
    """해상도가 목표와 일치하는지 검사"""
    return width == target_width and height == target_height


def check_codec(codec: str, supported_codecs: set[str] = SUPPORTED_CODECS) -> bool:
    """코덱이 지원되는지 검사"""
    return codec.strip() in supported_codecs


def check_fps(fps: float, min_fps: float = MIN_FPS) -> bool:
    """FPS가 최소 기준 이상인지 검사"""
    return fps > min_fps


def validate_or_raise(metadata: VideoMetadata) -> None:
    """전처리 전 검증. 처리 불가능한 영상은 에러 발생"""
    duration_status = check_duration(metadata.duration_seconds)
    if duration_status == -1:
        raise ValueError(f"영상이 너무 짧습니다: {metadata.duration_seconds}초")

    if not check_codec(metadata.codec):
        raise ValueError(f"지원하지 않는 코덱입니다: {metadata.codec}")

    if not check_fps(metadata.fps):
        raise ValueError(f"FPS가 너무 낮습니다: {metadata.fps}")
