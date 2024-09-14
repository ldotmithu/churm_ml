from mlProject import logging
from mlProject.utils.common import *
from mlProject.config.configuration import *

class ModelTrain:
    def __init__(self,config:ModelTrainConfig):
        self.config=config
        
    def preprocess_model(self):
        num_col=['CreditScore',
                                'Age',
                                'Tenure',
                                'Balance',
                                'NumOfProducts',
                                'HasCrCard',
                                'IsActiveMember',
                                'EstimatedSalary']    
        cat_col=['Geography', 'Gender']
        
        