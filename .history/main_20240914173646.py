from src.mlProject.pipeline.stage_01_data_ingestion import *
from src.mlProject.pipeline.stage_02_data_transfomation import *
from src.mlProject.pipeline.stage_03_model_train import *
from src.mlProject.pipeline.stage_04_model_evaluation import *
from src.mlProject import logging

Stage_name='Data Ingestion  stage'
try:
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    logging.info('Data Ingestion Completed')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('                                                                                ')
except Exception as e:
    logging.exception(e)
    raise e


Stage_name='Data Transfomation  stage'
try:
    data_transfomation=DataTransfomationPipeline()
    data_transfomation.main()
    logging.info('Data Transfomation Completed')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('                                                                                ')
except Exception as e:
    logging.exception(e)
    raise e


Stage_name='Model Training  stage'
try:
    model_training=ModelTrainPipeline()
    model_training.main()
    logging.info('Model Training Completed')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('                                                                                ')
except Exception as e:
    logging.exception(e)
    raise e

Stage_name='Model Evaluation  stage'
try:
    model_evaluation=ModelEvaluationPipeline()
    model_evaluation.main()
    logging.info('Model Evaluation Completed')
    logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('                                                                                ')
except Exception as e:
    logging.exception(e)
    raise e
