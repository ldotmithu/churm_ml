from mlProject.config.configuration import *
import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score



class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config
        
    def eval_metrics(self,actual,pred):
        acc_score=accuracy_score(actual,pred)
        
        return acc_score
    
    
    def save_metrics(self):
        
        
            