# ML Hub Backend Service

# Import FastAPI
from fastapi import FastAPI
from app.routers import user

# Create FastAPI instance
app = FastAPI()


# Define root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to ML Hub!"}


# Include user router
app.include_router(user.router)
