# routes for user management

from fastapi import APIRouter, status
from app.models.user import User

# create router instance for user management routes
router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

db_dict = {}


# user creation route
@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_user():
    return {"message": "User created"}


# get all users route
@router.get("/all", status_code=status.HTTP_200_OK)
async def get_all_users():
    return {"data": "All users"}


# get a user route
@router.get("/{user_id}", status_code=status.HTTP_200_OK)
async def get_user(user_id: int):
    return {"data": user_id}
