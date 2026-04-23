from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# load once (important for speed)
classifier = pipeline("sentiment-analysis")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment API running"}

@app.post("/predict")
async def predict(request: TextRequest):
    result = classifier(request.text)[0]
    return {
        "label": result["label"],
        "score": float(result["score"])
    }
