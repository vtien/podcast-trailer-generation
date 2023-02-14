from app.model import PodcastIntroExtractorModel
from fastapi import FastAPI
from pydantic import BaseModel

class PodcastTranscript(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    introduction: str

app = FastAPI()

@app.get("/")
async def root():
    return {"health_check": "OK"}    

@app.post("/predict", response_model=PredictionOutput)
async def pred(transcript: PodcastTranscript):

    model = PodcastIntroExtractorModel()
    pred = model.predict(transcript.text)
    return {"introduction": pred['prediction']}