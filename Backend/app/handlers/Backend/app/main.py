# ML Hub Backend Service

# Import FastAPI
from fastapi import FastAPI
from app.routers import user, auth, ml

# Import database functions
from app.db.database import create_db


# Create database tables
create_db()

# Create FastAPI instance
app = FastAPI()


# Define root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to ML Hub!"}


# Include user router
app.include_router(user.router)

# Include auth router
app.include_router(auth.router)

# Include ML router
app.include_router(ml.router)
