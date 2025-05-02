import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body, Response
from fastapi.encoders import jsonable_encoder
from survival_model import __version__ as model_version
from survival_model.predict import predict_death_event

from app import __version__, schemas
from app.config import settings

import prometheus_client as prom
from survival_model.config.core import config
from survival_model.processing.data_manager import load_dataset
from sklearn.metrics import accuracy_score, f1_score

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            'age': 65.0,
            'anaemia': 1,
            'creatinine_phosphokinase': 160,
            'diabetes': 1,
            'ejection_fraction': 20,
            'high_blood_pressure': 0,
            'platelets': 327000.00,
            'serum_creatinine': 27.00,
            'serum_sodium': 116,
            'sex': 0,
            'smoking': 0,
            'time': 8
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = predict_death_event(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results


test_data = load_dataset(file_name = config.app_config_.training_data_file)
f1_metric = prom.Gauge('patient_survival_f1_score', 'F1 score for random 100 test samples')
accuracy_metric = prom.Gauge('patient_survival_accuracy_score', 'Accuracy score for random 100 test samples')

def update_metrics(): 
    test = test_data.sample(100) 
    test_feat = test.drop(config.model_config_.target, axis=1) 
    test_target = test[config.model_config_.target].values
    test_pred = predict_death_event(input_data=test_feat)['predictions'] 
    accuracy = accuracy_score(test_target, test_pred).round(3) 
    accuracy_metric.set(accuracy)
    f1 = f1_score(test_target, test_pred, average='weighted').round(3)
    f1_metric.set(f1)


@api_router.get("/metrics")
async def get_metrics(): 
    update_metrics() 
    return Response(media_type="text/plain",
                    content = prom.generate_latest())