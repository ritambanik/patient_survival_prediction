import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]

print("Parent Directory:", root)

sys.path.append(str(root))

import pickle
from typing import Union
import numpy as np
import pandas as pd
import os

from survival_model.config.core import TRAINED_MODEL_DIR, config
from survival_model import __version__ as _version

def load_model():
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    model_path = TRAINED_MODEL_DIR / save_file_name
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    

# Load the model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False
    model = None
    
    
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes,
       ejection_fraction, high_blood_pressure, platelets,
       serum_creatinine, serum_sodium, sex, smoking, time) -> int:
    
    if not model_loaded:
        return "Model not loaded. Please check if the model file exists."
    
    """Make a prediction using a saved model """
    

    input_data = {
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking],
        'time': [time]
    }

    input_df = pd.DataFrame(input_data)
    
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        print("Prediction:", prediction)
        return f"${prediction:,.2f}"
    except Exception as e:
        return f"Error making prediction: {str(e)}"
        
        