from mlProject import logging
from mlProject.utils.common import *
from mlProject.config.configuration import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class ModelTrain:
    def __init__(self,config:ModelTrainConfig):
        self.config=config
        
    def preprocess_model(self):
        try:
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
                    ('num_pipeline',num_pipeline,num_col),
                    ('cat_pipeline',cat_pipeline,cat_col)
                ])
            return preprocess
        
        except Exception as e:
            logging.exception(e)
            raise e   


           