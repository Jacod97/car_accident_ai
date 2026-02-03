"""요청 정보 로깅"""

import logging

from src.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


def log_request_info(
    content: list[dict],
    frame_count: int,
    width: int,
    height: int,
) -> dict:
    """
    요청 정보 로깅 및 반환

    Returns:
        로깅 정보 dict
    """
    token_count = count_tokens(content)

    info = {
        "frame_count": frame_count,
        "resolution": f"{width}x{height}",
        "token_count": token_count,
    }

    logger.info(
        "[토큰 로그] 프레임: %d개, 해상도: %dx%d, 토큰: %s",
        frame_count, width, height, f"{token_count:,}"
    )

    return info
