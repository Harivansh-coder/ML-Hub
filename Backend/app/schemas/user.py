# script for the user model for the database

import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.db.database import Base


# create user model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    role = Column(String, default="user")
