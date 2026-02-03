"""분석기 추상 베이스 클래스 (Template Method 패턴)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable

from src.analysis.content_builder import build_content
from src.models import AnalysisResult


class BaseAnalyzer(ABC):
    """LLM 기반 영상 프레임 분석기의 추상 베이스 클래스.

    Template Method 패턴 적용:
    - _build_content()가 고정된 알고리즘 골격을 정의
    - 서브클래스는 4개의 추상 메서드를 오버라이딩하여 변하는 부분만 구현

    의존성 주입:
    - 모델명과 프롬프트 경로를 생성자로 주입 (모듈 레벨 전역 변수 제거)
    """

    def __init__(self, model_name: str, prompt_path: str | Path):
        self._model = ChatGoogleGenerativeAI(model=model_name)
        self._prompt_path = Path(prompt_path)

    # ---- 추상 메서드 (서브클래스 필수 구현) ----

    @abstractmethod
    def _load_prompt(self) -> str:
        """프롬프트 템플릿을 로드하고 변수를 포맷."""
        ...

    @abstractmethod
    def _build_timestamps(self, frames: list[np.ndarray], **kwargs) -> list[str]:
        """각 프레임에 대한 타임스탬프 라벨 생성."""
        ...

    @abstractmethod
    def _parse_response(self, text: str) -> AnalysisResult:
        """LLM 텍스트 출력을 타입화된 결과로 파싱."""
        ...

    @abstractmethod
    def _prepare_input(self, raw_input: Any) -> tuple[list[np.ndarray], dict]:
        """호출자의 입력을 (프레임 리스트, 추가 kwargs)로 정규화.

        Detector는 list[ndarray]를 받고, DetailAnalyzer는 dict를 받는 등
        입력 형태가 다르므로 이 메서드에서 통일.
        """
        ...

    # ---- Template Method (고정 알고리즘) ----

    def _build_content(self, raw_input: Any) -> list[dict]:
        """템플릿 메서드: 입력 정규화 → 프롬프트 로드 → 타임스탬프 생성 → content 빌드"""
        frames, kwargs = self._prepare_input(raw_input)
        prompt_text = self._load_prompt()
        timestamps = self._build_timestamps(frames, **kwargs)
        return build_content(prompt_text, frames, timestamps)

    # ---- 체인 조립 (오버라이딩 가능) ----

    def build_chain(self) -> Runnable:
        """LangChain 체인 생성. 서브클래스에서 오버라이딩하여 파이프라인 커스터마이징 가능."""
        return (
            RunnableLambda(self._build_content)
            | RunnableLambda(lambda content: [HumanMessage(content=content)])
            | self._model
            | StrOutputParser()
            | RunnableLambda(self._parse_response)
        )

    # ---- 고수준 API ----

    def analyze(self, input_data: Any) -> AnalysisResult:
        """전체 체인을 실행하고 타입화된 결과를 반환."""
        chain = self.build_chain()
        return chain.invoke(input_data)
