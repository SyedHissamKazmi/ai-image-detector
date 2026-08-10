from fastapi import FastAPI
from app.api.routes import router as analyze_router


app = FastAPI(
    title="AI Image Detector",
    description="Detect whether an image is likely AI-generated or human-created.",
    version="0.1.0",
)

app.include_router(analyze_router)

@app.get("/")
def root():
    return {
        "project": "AI Image Detector",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
