# file contains all ml models used by ml routes
from pydantic import BaseModel, Field


class MLRequest(BaseModel):
    text: str = Field(..., min_length=100, max_length=1000)


#  ML request model for the /ml/similar route
class SimilarityRequest(BaseModel):
    text1: str = Field(..., min_length=50, max_length=1000)
    text2: str = Field(..., min_length=50, max_length=1000)
