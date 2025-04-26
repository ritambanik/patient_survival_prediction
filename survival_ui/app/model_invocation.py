import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))

import pickle
from typing import Union
import numpy as np
import pandas as pd
import os


def load_model():
    model_path = "./patient_survival_model_output_v0.0.1.pkl"
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
    
    print("Loading model...")
    
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
        
        