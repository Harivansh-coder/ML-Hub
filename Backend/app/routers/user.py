# routes for user management

from fastapi import APIRouter, status, Depends, HTTPException
from app.models.user import UserRequest, UserResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.user import User
from app.utils.verify_hash import hash_password
from app.utils.oauth2 import get_current_user

# create router instance for user management routes
router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


# user creation route
@router.post(
    "/create", status_code=status.HTTP_201_CREATED, response_model=UserResponse
)
async def create_user(user: UserRequest, db: Session = Depends(get_db)):
    # check if user already exists
    user_already_exists = (
        db.query(User).filter(User.email == user.email).first() is not None
    )

    # raise exception if user already exists
    if user_already_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="User already exists"
        )

    # hash user password
    user.password = hash_password(user.password)

    # create new user
    new_user = User(**user.dict())
    # add new user to database
    db.add(new_user)
    # commit changes to database
    db.commit()
    db.refresh(new_user)
    return new_user


# get all users route
@router.get("/all", status_code=status.HTTP_200_OK, response_model=list[UserResponse])
async def get_all_users(
    db: Session = Depends(get_db), _: int = Depends(get_current_user)
):
    return db.query(User).all()


# get a user route
@router.get("/", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def get_user(
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user),
):
    # get user from database
    user = db.query(User).filter(User.id == current_user.id).first()

    # raise exception if user does not exist
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist"
        )

    return user


# update a user route
@router.put("/", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def update_user(
    user: UserRequest,
    db: Session = Depends(get_db),
    current_user: int = Depends(get_current_user),
):
    # get user from database
    user_to_update = db.query(User).filter(User.id == current_user.id).first()

    # raise exception if user does not exist
    if user_to_update is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist"
        )

    # update user details
    user_to_update.username = user.username
    user_to_update.email = user.email
    user_to_update.password = user.password

    # commit changes to database
    db.commit()
    db.refresh(user_to_update)
    return user_to_update
