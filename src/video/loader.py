import cv2
from pathlib import Path

from src.models import VideoMetadata


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

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        duration_seconds = total_frames / fps if fps > 0 else 0
        minutes, seconds = divmod(int(duration_seconds), 60)
        duration_formatted = f"{minutes:02d}:{seconds:02d}"

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
