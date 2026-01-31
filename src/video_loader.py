import cv2
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    """영상 메타데이터를 저장하는 데이터 클래스"""
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


class VideoLoader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def load(self) -> cv2.VideoCapture:
        """영상 파일을 로드하고 VideoCapture 객체를 반환합니다."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {self.video_path}")

        return self._cap

    def get_metadata(self) -> VideoMetadata:
        """영상의 메타데이터를 추출하여 반환합니다."""
        if self._cap is None:
            self.load()

        cap = self._cap

        # 기본 영상 속성 추출
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        fps = cap.get(cv2.CAP_PROP_FPS)  
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  

        # 코덱 정보 추출 (fourcc 코드를 문자열로 변환)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        # 영상 길이 계산
        duration_seconds = total_frames / fps if fps > 0 else 0
        minutes, seconds = divmod(int(duration_seconds), 60)
        duration_formatted = f"{minutes:02d}:{seconds:02d}"

        # 파일 크기 (MB 단위)
        file_size_mb = self.video_path.stat().st_size / (1024 * 1024)

        self._metadata = VideoMetadata(
            file_path=str(self.video_path),
            file_name=self.video_path.name,
            file_size_mb=round(file_size_mb, 2),
            duration_seconds=round(duration_seconds, 2),
            duration_formatted=duration_formatted,
            width=width,
            height=height,
            fps=round(fps, 2),
            total_frames=total_frames,
            codec=codec,
        )
        return self._metadata


if __name__ == "__main__":
    # 테스트 실행: data/mp4 폴더의 영상 메타데이터 출력
    video_path = r"C:\Project\toy\car_accident_ai\data\mp4\bb_1_130209_vehicle_45_043.mp4"
    loader = VideoLoader(video_path)

    meta = loader.get_metadata()

    print("file_size_mb:" ,meta.file_size_mb)
    print("duration_seconds:" ,meta.duration_seconds)
    print("duration_formatted:" ,meta.duration_formatted)
    print("width:" ,meta.width)
    print("height:" ,meta.height)
    print("fps:" ,meta.fps)
    print("total_frames:" ,meta.total_frames)
    print("codec:" ,meta.codec)