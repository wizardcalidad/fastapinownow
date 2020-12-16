import joblib
import sklearn
import numpy as np

from utils import clean_text, CleanTextTransformer

from pydantic.main import BaseModel

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

models = {
    "complement": {
        "tfidf": joblib.load("tweets_complement_with_tfidf_vectorizer.joblib"),
    }
}


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    output: str


@app.get("/ping")
def ping():
    return Response(content="pong", media_type="text/plain")


@app.post('/predict', response_model=PredictResponse)
def predict(parameters: PredictRequest):
    text = parameters.text

    x = [text]
    y = models["complement"]["tfidf"].predict(x)
    y = True if y == 'positive' else False

    response = {"output": "positive" if y else "negative"}

    return response


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
