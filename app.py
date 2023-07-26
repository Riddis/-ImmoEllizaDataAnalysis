from fastapi import FastAPI
from pydantic import BaseModel
import json
from src.preprocessing import preprocess_new_data
from xgboost import XGBRegressor

# uvicorn main:app --reload

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "POST requirements"}

@app.post("/predict")
async def predict(json_data):
    df, status = preprocess_new_data(json_data)

    if status != 200:
        for error in df:
            print(error.message)
        return
    
    model = XGBRegressor.load_model('models/xgbmodel.model')
    prediction = model.predict(df)

    dict = {"prediction": prediction,
            "status_code": status}

    return json.dumps(dict) 
            