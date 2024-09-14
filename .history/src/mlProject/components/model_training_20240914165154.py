from mlProject import logging
from mlProject.utils.common import *
from mlProject.config.configuration import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd 
from imblearn.combine import SMOTETomek, SMOTEENN

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

    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)
        
        smt = SMOTEENN(random_state=42,sampling_strategy='minority' )
        # Fit the model to generate the data.
        train_data, test_data = smt.fit_resample(train_data, test_data)
        
        target_col='Exited'
        
        X_train=train_data.drop(columns=['RowNumber','CustomerId','Surname','Exited'],axis=1)
        X_test=test_data.drop(columns=['RowNumber','CustomerId','Surname','Exited'],axis=1)
        y_train=train_data[target_col]
        y_test=test_data[target_col]
        
        preprocess_obj=self.preprocess_model()
        
        X_train=preprocess_obj.fit_transform(X_train)
        X_test=preprocess_obj.transform(X_test)
        
        
        
           