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
        num_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scale',StandardScaler())
            ])
        cat_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one',OneHotEncoder())
            ])                      
        preprocess=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipeline',cat_pipeline,cat_col)
            ])


           