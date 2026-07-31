from collections.abc import AsyncGenerator

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# ---------------------------------------------------------------------------
# Engine — ssl=require is already embedded in the DATABASE_URL from .env
# ---------------------------------------------------------------------------
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,            # set True locally to debug SQL; never in production logs
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,    # recycle stale connections transparently
    pool_recycle=1800,     # recycle connections every 30 min (AWS RDS idle timeout)
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async DB session and guarantees
    rollback on error and session close in all cases.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError:
            await session.rollback()
            raise
        finally:
            await session.close()
