from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import narrative, data_source, users
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(users.router, prefix="/authorization", tags=["Login & Signup"])
app.include_router(narrative.router, prefix="/narrative", tags=["narrative"])
app.include_router(data_source.router, prefix="/data-source", tags=["data source"])

@app.get("/")
async def root():
    return {"message": "Welcome to Othor API"}