from src.config import TARGET_WIDTH, TARGET_HEIGHT, MAX_VIDEO_SEC


# OpenCV 지원 코덱 목록
SUPPORTED_CODECS = {
    "avc1", "h264", "H264",  # H.264
    "hevc", "hev1", "H265",  # H.265/HEVC
    "mp4v", "MP4V",          # MPEG-4
    "XVID", "xvid",          # Xvid
    "DIVX", "divx",          # DivX
    "MJPG", "mjpg",          # Motion JPEG
    "VP80", "VP90",          # VP8, VP9
    "WMV1", "WMV2", "WMV3",  # Windows Media Video
    "FMP4",                  # FFmpeg MPEG-4
}


def check_duration(duration_seconds: float, min_sec: float = 10.0, max_sec: float = MAX_VIDEO_SEC) -> int:
    """영상 길이 검사"""
    if duration_seconds < min_sec:
        return -1  # 짧
    elif duration_seconds > max_sec:
        return 1  # 김
    else:
        return 0  # 적절


def check_resolution(width: int, height: int, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT) -> bool:
    """해상도가 목표와 일치하는지 검사"""
    return width == target_width and height == target_height


def check_codec(codec: str, supported_codecs: set[str] = SUPPORTED_CODECS) -> bool:
    """코덱이 OpenCV에서 지원되는지 검사"""
    return codec.strip() in supported_codecs


def check_fps(fps: float, min_fps: float = 10.0) -> bool:
    """FPS가 최소 기준 이상인지 검사"""
    return fps > min_fps
