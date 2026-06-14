"""
Database setup with SQLAlchemy.
Provides session management and base model class.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from backend.config import get_settings


settings = get_settings()

# Create engine
engine = create_engine(
    settings.database_url,
    echo=settings.app_debug,  # Log SQL queries in debug mode
    pool_pre_ping=True,       # Verify connections before use
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def get_db():
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once on startup."""
    Base.metadata.create_all(bind=engine)
