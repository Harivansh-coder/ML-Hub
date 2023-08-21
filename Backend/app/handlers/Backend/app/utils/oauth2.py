# OAUTH2 implementation for the application
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from ..db.database import get_db
from ..schemas.user import User
from ..models.token import TokenData

# create oauth2_scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# jwt token settings
SECRET_KEY = "c0b0f0b0-0f0b-0f0b-c0b0-c0b0f0b0c0b0"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# create access token function to create jwt access token
def create_access_token(data: dict):
    # create jwt payload
    to_encode = data.copy()
    # get current time
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # add expire time to payload
    to_encode.update({"exp": expire})
    # encode payload
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


# create decode token function to decode jwt token
def decode_token(token: str):
    # decode token
    decoded_token_playload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)

    # return decoded token payload
    return decoded_token_playload


# create verify token function to verify jwt token
def verify_token(token: str):
    # create credentials exception
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # verify token
    try:
        decoded_token = decode_token(token)
        # get user id from token
        user_id = decoded_token.get("user_id")

        if user_id is None:
            raise credentials_exception

        token_data = TokenData(user_id=user_id)

    except JWTError:
        raise credentials_exception

    # return token data
    return token_data


# get the current user from the token
def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    token_data = verify_token(token)

    current_user = db.query(User).filter(User.id == token_data.user_id).first()

    return current_user
