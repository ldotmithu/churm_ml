from mlProject.config.configuration import *
from mlProject import logging
import os,zipfile
import urllib.request



class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
        
    def download_file(self):
            