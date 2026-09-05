from sqlmodel import SQLModel, create_engine, Session
import os
from pathlib import Path

# Render's container has a writable /data directory. A relative SQLite path
# such as ./data/db.sqlite points to /app/data, which is not created by the
# image and makes SQLAlchemy fail during startup.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////data/db.sqlite")

if DATABASE_URL.startswith("sqlite:///"):
    sqlite_path = DATABASE_URL.removeprefix("sqlite:///")
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
