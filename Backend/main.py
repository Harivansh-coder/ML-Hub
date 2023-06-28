# ML Hub Backend Service

# Import FastAPI
from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()


# Define root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}
