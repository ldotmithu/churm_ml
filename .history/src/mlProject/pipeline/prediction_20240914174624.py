import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject.utils.common import load_object


class PredictionPipeline:
    def __init__(self):
        self.preprocessor = load_object('arifacts/model_training/preprocess.pkl')
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    
    def predict(self, data):
        data = self.preprocessor.transform(data)
        prediction = self.model.predict(data)

        return prediction