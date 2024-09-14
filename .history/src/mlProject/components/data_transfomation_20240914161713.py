from mlProject import logging
import pandas as pd 
from mlProject.config.configuration import *
from sklearn.model_selection import train_test_split


class DataTransfomation:
    def __init__(self,config:DataTransfomationConfig):
        self.config=config
        
    def split_data(self):
        data=pd.read_csv(self.config.data_path)
        logging.info('Read the data through pandas read_csv')
        
        train_data,test_data=train_test_split(data,test_size=0.2)
        
        train_data.to_csv(os.path.join(self.config.root_dir,'train.csv'),index=False)
        test_data.to_csv(os.path.join(self.config.root_dir,'test.csv'),index=False)
        
    