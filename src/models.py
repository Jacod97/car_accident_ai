from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict


@dataclass
class VideoMetadata:
    """영상 메타데이터"""
    file_path: str
    file_name: str
    file_size_mb: float
    duration_seconds: float
    duration_formatted: str
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str


@dataclass
class AnalysisResult(ABC):
    """분석 결과 추상 베이스 클래스"""
    raw_response: str

    @abstractmethod
    def has_data(self) -> bool:
        """유의미한 분석 데이터가 있는지 여부"""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """사람이 읽을 수 있는 포맷"""
        ...

    def to_dict(self) -> dict:
        """직렬화. 서브클래스에서 오버라이딩 가능."""
        return asdict(self)


@dataclass
class DetectorResult(AnalysisResult):
    """1차 분석 결과"""
    descriptions: dict[int, str] = None
    accident_range: list[int] | None = None

    def has_data(self) -> bool:
        return bool(self.descriptions)

    def __str__(self) -> str:
        lines = [f"{sec}초: {desc}" for sec, desc in sorted(self.descriptions.items())]
        range_str = (
            f"{self.accident_range[0]}~{self.accident_range[1]}초"
            if self.accident_range else "없음"
        )
        lines.append(f"사고 구간: {range_str}")
        return "\n".join(lines)


@dataclass
class DetailResult(AnalysisResult):
    """2차 상세 분석 결과"""
    details: dict[float, str] = None
    summary: str = ""

    def has_data(self) -> bool:
        return bool(self.details)

    def __str__(self) -> str:
        lines = [f"{ts:.1f}초: {desc}" for ts, desc in sorted(self.details.items())]
        lines.append(f"사고 요약: {self.summary}")
        return "\n".join(lines)
