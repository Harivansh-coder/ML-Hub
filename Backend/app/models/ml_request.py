# file contains all ml models used by ml routes
from pydantic import BaseModel, Field


class MLRequest(BaseModel):
    text: str = Field(..., min_length=100, max_length=1000)
