# user model for request body and response

from pydantic import BaseModel, EmailStr, constr
from enum import Enum
from datetime import datetime


# user role enum
class UserRole(Enum):
    user = "user"
    admin = "admin"


# user model for user signup response
class UserResponse(BaseModel):
    id: int
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    # now the created_at field which is a datetime object is converted to a string
    created_at: datetime

    class Config:
        orm_mode = True


# user model for user signup
class UserRequest(BaseModel):
    username: constr(min_length=3, max_length=50)
    email: EmailStr
    password: constr(min_length=8, max_length=50)


# user model for user login
class UserLogin(BaseModel):
    email: constr(min_length=3, max_length=50)
    password: constr(min_length=8, max_length=50)
