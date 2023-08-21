# database connection to posgres database

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# database connection string
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:harry1234@localhost:5432/postgres"

# create database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# create base class
Base = declarative_base()


# get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# create function to create database tables
def create_db():
    Base.metadata.create_all(bind=engine)
