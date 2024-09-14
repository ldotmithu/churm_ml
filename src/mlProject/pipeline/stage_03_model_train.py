from mlProject.components.model_training import *
from mlProject import logging
from mlProject.config.configuration import *

class ModelTrainPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config=ConfigurationManager()
        model_training_config=config.get_model_training_config()
        model_training=ModelTrain(config=model_training_config)
        model_training.preprocess_model()
        model_training.train()