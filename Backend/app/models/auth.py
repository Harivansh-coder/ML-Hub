# OAUTH Customizations for the login route in the backend

from fastapi.param_functions import Form
from pydantic import EmailStr


# Custom OAuth2PasswordRequestForm to accept email instead of username
class EmailPasswordOAuth2PasswordRequestForm:
    def __init__(
        self,
        grant_type: str = Form(default=None, regex="password"),
        email: EmailStr = Form(),
        password: str = Form(),
    ):
        self.grant_type = grant_type
        self.email = email
        self.password = password
