import joblib
import sklearn
import numpy as np

from utils import clean_text, CleanTextTransformer

from pydantic.main import BaseModel

from fastapi import FastAPI
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import JSONResponse
from fastapi import FastAPI, Response

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
