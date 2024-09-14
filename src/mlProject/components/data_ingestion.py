from mlProject.config.configuration import *
from mlProject import logging
import os,zipfile
import urllib.request



class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_path):
            urllib.request.urlretrieve(self.config.URL,self.config.local_data_path)
            logging.info('Zip Data Dpwnloaded')
            
        else:
            logging.info('file Already exists') 
            
    def Exreact_all(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_path,'r') as file:
            file.extractall(unzip_path)
            logging.info('Extrcat File Successfull')         
