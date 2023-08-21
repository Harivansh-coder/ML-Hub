# this file contains the script for the authentication of the user

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.token import Token
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.user import User
from app.models.auth import EmailPasswordOAuth2PasswordRequestForm
from app.utils.verify_hash import verify_password
from app.utils.oauth2 import create_access_token

# router for the authentication of the user
router = APIRouter(
    prefix="/auth",
    tags=["authencation"],
    responses={404: {"description": "Not found"}},
)


# login route for the user
@router.post("/login", status_code=status.HTTP_200_OK, response_model=Token)
async def login_user(
    user_credentials: EmailPasswordOAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # get user from database
    user = db.query(User).filter(User.email == user_credentials.email).first()

    # check if user exists
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Email not registered"
        )

    # check if password is correct
    if not verify_password(user_credentials.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password"
        )

    # create access token
    access_token = create_access_token(data={"user_id": user.id})

    # return access token
    return {"access_token": access_token, "token_type": "bearer"}
