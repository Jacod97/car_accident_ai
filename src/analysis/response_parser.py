"""LLM 응답 파싱"""

from src.models import DetectorResult, DetailResult


def parse_detector_response(text: str) -> DetectorResult:
    """1차 분석 응답 파싱"""
    descriptions = {}
    accident_range = None

    lines = text.strip().split("\n")
    in_description = False
    in_accident = False
    start = None
    end = None

    for line in lines:
        line = line.strip()

        if "[상황 설명]" in line:
            in_description = True
            in_accident = False
            continue

        if "[사고 구간]" in line:
            in_description = False
            in_accident = True
            continue

        if in_description and "초:" in line:
            parts = line.split("초:", 1)
            if len(parts) == 2:
                sec = parts[0].strip()
                if sec.isdigit():
                    descriptions[int(sec)] = parts[1].strip()

        if in_accident:
            if line.startswith("시작:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    start = int(val)
            elif line.startswith("종료:"):
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    end = int(val)

    if start is not None and end is not None:
        accident_range = [start, end]

    return DetectorResult(
        descriptions=descriptions,
        accident_range=accident_range,
        raw_response=text,
    )


def parse_detail_response(text: str) -> DetailResult:
    """2차 상세 분석 응답 파싱"""
    details = {}
    summary_lines = []

    lines = text.strip().split("\n")
    in_detail = False
    in_summary = False

    for line in lines:
        line = line.strip()

        if "[상세 분석]" in line:
            in_detail = True
            in_summary = False
            continue

        if "[사고 요약]" in line:
            in_detail = False
            in_summary = True
            continue

        if in_detail and "초:" in line:
            parts = line.split("초:", 1)
            if len(parts) == 2:
                try:
                    timestamp = float(parts[0].strip())
                    details[timestamp] = parts[1].strip()
                except ValueError:
                    pass

        if in_summary and line:
            summary_lines.append(line)

    return DetailResult(
        details=details,
        summary=" ".join(summary_lines),
        raw_response=text,
    )
