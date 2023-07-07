# routes for user management

from fastapi import APIRouter, status, Depends, HTTPException
from app.models.user import UserRequest
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.user import User

# create router instance for user management routes
router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


# user creation route
@router.post("/create", status_code=status.HTTP_201_CREATED)
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

    # create new user
    new_user = User(**user.dict())
    # add new user to database
    db.add(new_user)
    # commit changes to database
    db.commit()
    db.refresh(new_user)
    return {"data": new_user}


# get all users route
@router.get("/all", status_code=status.HTTP_200_OK)
async def get_all_users(db: Session = Depends(get_db)):
    return {"data": db.query(User).all()}


# get a user route
@router.get("/{user_id}", status_code=status.HTTP_200_OK)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    # get user from database
    user = db.query(User).filter(User.id == user_id).first()

    # raise exception if user does not exist
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist"
        )

    return {"data": user}


# update a user route
@router.put("/{user_id}", status_code=status.HTTP_200_OK)
async def update_user(user_id: int, user: UserRequest, db: Session = Depends(get_db)):
    # get user from database
    user_to_update = db.query(User).filter(User.id == user_id).first()

    # raise exception if user does not exist
    if user_to_update is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist"
        )

    # update user details
    user_to_update.username = user.username
    user_to_update.email = user.email
    user_to_update.password = user.password
    user_to_update.role = user.role

    # commit changes to database
    db.commit()
    db.refresh(user_to_update)
    return {"data": user_to_update}
