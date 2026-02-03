"""
설정 모듈

토큰 사용량 최적화를 위한 설정값
"""

# 해상도 설정 (낮출수록 토큰 절약)
TARGET_WIDTH = 512
TARGET_HEIGHT = 288

# 1차 분석: 프레임 추출 간격 (초)
# 1.0 = 1초당 1프레임
DETECTOR_INTERVAL_SEC = 1.0

# 2차 분석: FPS (낮출수록 토큰 절약)
# 5 = 0.2초 간격, 3 = 0.33초 간격
DETAIL_FPS = 3

# JPEG 품질 (낮출수록 토큰 절약, 50~85 권장)
JPEG_QUALITY = 70

# 영상 길이 제한 (초)
MAX_VIDEO_SEC = 20.0
MIN_VIDEO_SEC = 10.0

# FPS 최소 기준
MIN_FPS = 10.0

# Gemini 모델
GEMINI_MODEL = "gemini-2.5-flash"

# 사고 구간 전후 패딩 (초)
ACCIDENT_PADDING_SEC = 1
