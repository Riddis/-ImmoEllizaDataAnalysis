from fastapi import FastAPI
import json
from src.preprocessing import preprocess_new_data
from xgboost import XGBRegressor
import joblib

# uvicorn app:app --reload

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "POST requirements"}

@app.post("/predict")
async def predict(json_data: dict):
    encoder = joblib.load("models/encoder.save")
    scaler = joblib.load("models/scaler.save")
    print("Preprocessing data")
    df, status = preprocess_new_data(json_data, encoder, scaler)

    if status != 200:
        print("Preproccesing failed")
        for error in df:
            print(error.message)
        return
    
    print("Predicting")
    model = XGBRegressor.load_model('models/xgbmodel.model')
    prediction = model.predict(df)

    dict = {"prediction": prediction,
            "status_code": status}

    return json.dumps(dict) 
            