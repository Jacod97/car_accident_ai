"""사고 감지 AI (1차 분석)

BaseAnalyzer를 상속하여 4개 추상 메서드를 오버라이딩:
- _load_prompt: detector.prompt 로드
- _build_timestamps: 정수 초 기반 라벨
- _parse_response: DetectorResult 반환
- _prepare_input: list[ndarray] → (frames, {})
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.config import DETECTOR_INTERVAL_SEC, GEMINI_MODEL
from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.response_parser import parse_detector_response
from src.models import DetectorResult


load_dotenv()

_DEFAULT_PROMPT = Path(__file__).parent.parent / "prompts" / "detector.prompt"


class AccidentDetector(BaseAnalyzer):
    """1차 분석: 간격 기반 프레임 분석으로 사고 감지.

    BaseAnalyzer의 4개 추상 메서드를 오버라이딩하여
    사고 감지에 특화된 동작을 구현.
    """

    def __init__(
        self,
        model_name: str = GEMINI_MODEL,
        prompt_path: str | Path = _DEFAULT_PROMPT,
        interval_sec: float = DETECTOR_INTERVAL_SEC,
    ):
        super().__init__(model_name, prompt_path)
        self._interval_sec = interval_sec

    def _load_prompt(self) -> str:
        text = self._prompt_path.read_text(encoding="utf-8")
        return text.format(interval_sec=self._interval_sec)

    def _build_timestamps(self, frames: list[np.ndarray], **kwargs) -> list[str]:
        return [
            f"[프레임 {int((i + 1) * self._interval_sec)}초]"
            for i in range(len(frames))
        ]

    def _parse_response(self, text: str) -> DetectorResult:
        return parse_detector_response(text)

    def _prepare_input(self, raw_input: Any) -> tuple[list[np.ndarray], dict]:
        return raw_input, {}


# Backward-compatible
_detector = AccidentDetector()
accident_detector_chain = _detector.build_chain()
