# all machine learning related routes are here

# import dependencies
from fastapi import APIRouter
from app.handlers.summary import generate_summary
from app.models.ml_request import MLRequest

# create router instance for machine learning routes
router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    responses={404: {"description": "Not found"}},
)


# machine learning routes
@router.post("/summarize")
def summarize_text(text: MLRequest):
    # summarize text using extractive summarization

    return {"summary": generate_summary(text.text)}


# AI content detection
@router.get("/detect/{text}")
def detect_content(text: str):
    # detect content using BERT

    return {"message": "detected content"}


# AI content generation
@router.get("/generate/{text}")
def generate_content(text: str):
    # generate content using GPT-2

    return {"message": "generated content"}


# AI rephrasing text
@router.post("/rephrase/{text}")
def rephrase_text(text: str):
    # rephrase text using GPT-2

    return {"message": "rephrased text"}
