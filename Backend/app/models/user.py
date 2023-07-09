# user model for request body and response

from pydantic import BaseModel, EmailStr, constr
from enum import Enum


class UserRole(Enum):
    user = "user"
    admin = "admin"


# user model for user signup response
class UserResponse(BaseModel):
    id: int
    username: constr(min_length=3, max_length=50)
    email: EmailStr


# user model for user signup
class UserRequest(UserResponse):
    password: constr(min_length=8, max_length=50)


# user model for user login
class UserLogin(BaseModel):
    email: constr(min_length=3, max_length=50)
    password: constr(min_length=8, max_length=50)
