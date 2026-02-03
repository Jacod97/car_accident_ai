"""사고 상세 분석 AI (2차 분석)

BaseAnalyzer를 상속하여 4개 추상 메서드를 오버라이딩:
- _load_prompt: detail_analyzer.prompt 로드
- _build_timestamps: 소수점 초 기반 라벨
- _parse_response: DetailResult 반환
- _prepare_input: dict → (frames, {"start_sec": float})
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.config import DETAIL_FPS, GEMINI_MODEL
from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.response_parser import parse_detail_response
from src.models import DetailResult


load_dotenv()

_DEFAULT_PROMPT = Path(__file__).parent.parent / "prompts" / "detail_analyzer.prompt"


class AccidentDetailAnalyzer(BaseAnalyzer):
    """2차 분석: 사고 구간 고FPS 상세 분석.

    BaseAnalyzer의 4개 추상 메서드를 오버라이딩하여
    상세 분석에 특화된 동작을 구현.
    """

    def __init__(
        self,
        model_name: str = GEMINI_MODEL,
        prompt_path: str | Path = _DEFAULT_PROMPT,
        detail_fps: int = DETAIL_FPS,
    ):
        super().__init__(model_name, prompt_path)
        self._detail_fps = detail_fps
        self._detail_interval = round(1.0 / detail_fps, 2)

    def _load_prompt(self) -> str:
        text = self._prompt_path.read_text(encoding="utf-8")
        return text.format(detail_interval=self._detail_interval)

    def _build_timestamps(self, frames: list[np.ndarray], **kwargs) -> list[str]:
        start_sec = kwargs.get("start_sec", 0.0)
        return [
            f"[{start_sec + (i * self._detail_interval):.1f}초]"
            for i in range(len(frames))
        ]

    def _parse_response(self, text: str) -> DetailResult:
        return parse_detail_response(text)

    def _prepare_input(self, raw_input: Any) -> tuple[list[np.ndarray], dict]:
        return raw_input["frames"], {"start_sec": raw_input["start_sec"]}


# Backward-compatible
_detail_analyzer = AccidentDetailAnalyzer()
accident_detail_chain = _detail_analyzer.build_chain()
