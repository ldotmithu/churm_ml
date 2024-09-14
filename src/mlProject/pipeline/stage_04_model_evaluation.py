
from mlProject.components.model_evaluation import *
from mlProject import logging
from mlProject.config.configuration import *

class ModelEvaluationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_evaluation=ModelEvaluation(config=model_evaluation_config)
        model_evaluation.save_metrics()
        