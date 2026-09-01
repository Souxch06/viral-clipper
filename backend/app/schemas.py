from pydantic import BaseModel
from typing import Optional, List

class CreateJob(BaseModel):
    url: str

class JobStatus(BaseModel):
    id: int
    status: str
    step: Optional[str]
    storage_path: Optional[str]
    metadata: Optional[dict]
