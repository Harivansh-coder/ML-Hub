# all machine learning related routes are here

# import dependencies
from fastapi import APIRouter
from app.handlers.summary import generate_summary
from app.handlers.similarity import check_text_similarity
from app.models.ml_request import MLRequest, SimilarityRequest
from app.handlers.detect import predict

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
@router.post("/detect")
def detect_content(text: MLRequest):
    # detect content using BERT

    content_type = "true" if predict(text.text) == 1 else "false"
    return {"AI content detected": content_type}


# AI content generation
@router.get("/generate")
def generate_content(text: MLRequest):
    # generate content using GPT-2

    return {"message": "generated content"}


# AI rephrasing text
@router.post("/rephrase")
def rephrase_text(text: MLRequest):
    # rephrase text using GPT-2

    return {"message": "rephrased text"}


# AI similarity check
@router.post("/similarity")
def check_similarity(playload: SimilarityRequest):
    # check similarity between two texts using BERT
    similarity_scaore = check_text_similarity(playload.text1, playload.text2)
    if similarity_scaore > 0.8:
        return {"message": "similar content"}
    else:
        return {"message": "not similar content"}
