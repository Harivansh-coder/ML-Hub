# database connection and management script

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# database connection string
SQLALCHEMY_DATABASE_URL = "sqlite:///./app/db/mlhub.db"

# create database engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# create base class
Base = declarative_base()


# create function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# create function to create database tables
def create_db():
    Base.metadata.create_all(bind=engine)


# create function to drop database tables
def drop_db():
    Base.metadata.drop_all(bind=engine)


# create function to reset database
def reset_db():
    drop_db()
    create_db()
