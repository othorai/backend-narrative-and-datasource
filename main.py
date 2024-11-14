from fastapi import FastAPI
from routers import narrative
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(narrative.router, prefix="/narrative", tags=["narrative"])

@app.get("/")
async def root():
    return {"message": "Welcome to Othor API"}