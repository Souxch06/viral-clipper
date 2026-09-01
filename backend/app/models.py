from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
import json

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    original_url: Optional[str] = None
    status: str = "pending"   # pending / processing / done / error
    step: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    storage_path: Optional[str] = None
    metadata: Optional[str] = None  # JSON string: paths, transcript, clips, etc.

    def set_metadata(self, obj: dict):
        self.metadata = json.dumps(obj)

    def get_metadata(self):
        if self.metadata:
            return json.loads(self.metadata)
        return {}
