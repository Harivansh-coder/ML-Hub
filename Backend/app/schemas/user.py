# script for the user model for the database

import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.db.database import Base


# create user model
class User(Base):
    __tablename__ = "users"

    # auto generate id for each user
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50))
    email = Column(String(50), unique=True, index=True)
    password = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    role = Column(String(50), default="user")
