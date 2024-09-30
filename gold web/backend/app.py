from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import sklearn

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open(r'C:/Users/Yogesh E/OneDrive/Desktop/gold web/backend/gold_price_pred_model.pkl', 'rb'))

class GoldPriceModel(BaseModel):
    spxFloat: float
    usoFloat: float
    slvFloat: float
    euroUSDRatioFloat: float

@app.get('/')
def welcome_msg():
    return {
        'success': True,
        'message': 'Server of gold price prediction API is up and running successfully'
    }

@app.post('/pred-gold-price')
async def pred_price(values_from_frontend: GoldPriceModel):
    spx = values_from_frontend.spxFloat
    uso = values_from_frontend.usoFloat
    slv = values_from_frontend.slvFloat
    eurousdratio = values_from_frontend.euroUSDRatioFloat

    pred_data = pd.DataFrame([[spx, uso, slv, eurousdratio]], columns=['SPX', 'USO', 'SLV', 'EUR/USD'])
    prediction = model.predict(pred_data)

    return {
        'success': True,
        'pred_result_value': float(prediction[0])
    }
