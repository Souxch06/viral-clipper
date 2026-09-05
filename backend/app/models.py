from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text
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
    # ``metadata`` is reserved by SQLAlchemy's Declarative API. Keep the
    # existing database column name for compatibility, but use a safe Python
    # attribute name on the model.
    metadata_json: Optional[str] = Field(
        default=None,
        sa_column=Column("metadata", Text, nullable=True),
    )

    def set_metadata(self, obj: dict):
        self.metadata_json = json.dumps(obj)

    def get_metadata(self):
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}
