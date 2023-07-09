# this file contains the script for the authentication of the user

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import UserLogin
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.user import User

# router for the authentication of the user
router = APIRouter(
    prefix="/auth/login",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)


# login route for the user
@router.post("/", status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    # get user from database
    user = (
        db.query(User)
        .filter(User.email == user.email, User.password == user.password)
        .first()
    )

    # raise exception if user does not exist
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist"
        )

    return {"data": user}
