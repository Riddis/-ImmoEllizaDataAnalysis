from fastapi import FastAPI
import json
from src.preprocessing import preprocess_new_data
from xgboost import XGBRegressor
import joblib

# uvicorn app:app --reload

app = FastAPI()

@app.get('/')
async def root():
    return {'message': r"""'Please provide a JSON file with the following format:'
            'data': {
                'area': int,
                'property-type': 'APARTMENT' | 'HOUSE' | 'OTHERS',
                'rooms-number': int,
                'zip-code': int,
                'land-area': Optional[int],
                'garden': Optional[bool],
                'garden-area': Optional[int],
                'equipped-kitchen': Optional[bool],
                'province': Optional[
                'Antwerp' | 'Brussels' | 'East Flanders' | 'West Flanders' | 'Flemish Brabant' | 
                'Hainaut' |'Liège' | 'Limburg' | 'Luxembourg' | 'Namur' | 'Waloon Brabant'
                ],
                'swimming-pool': Optional[bool],
                'furnished': Optional[bool],
                'open-fire': Optional[bool],
                'terrace': Optional[bool],
                'terrace-area': Optional[int],
                'facades-number': Optional[int],
                'building-state': Optional[
                'NEW' | 'GOOD' | 'TO RENOVATE' | 'JUST RENOVATED' | 'TO REBUILD'
                ]
            }"""
    }

@app.post('/predict')
async def predict(json_data: dict):
    encoder = joblib.load('models/encoder.save')
    scaler = joblib.load('models/scaler.save')
    print('Preprocessing data')
    df, status = preprocess_new_data(json_data, encoder, scaler)

    if status != 200:
        dict = {'error': str(df.message),
                'status_code': status}
        return json.dumps(dict) 
    
    print('Predicting')
    model = XGBRegressor()
    model.load_model('models/xgbmodel.model')
    prediction = model.predict(df)

    dict = {'prediction': int(prediction[0]),
            'status_code': status}

    return json.dumps(dict) 
            