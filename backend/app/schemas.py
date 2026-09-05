from pydantic import BaseModel
from typing import Optional, List

class CreateJob(BaseModel):
    url: str
    max_clips: int = 4
    clip_duration: int = 20
    quality: str = "720"
    captions: bool = True
    zoom: bool = True
    caption_style: str = "bold"
    caption_position: str = "bottom"
    caption_size: int = 8
    aspect: str = "9:16"
    fps: int = 30
    bitrate: str = "auto"

class JobStatus(BaseModel):
    id: int
    status: str
    step: Optional[str]
    storage_path: Optional[str]
    metadata: Optional[dict]
