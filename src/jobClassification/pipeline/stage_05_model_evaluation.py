from jobClassification.logging import logger
from jobClassification.config.configuration import ConfigurationManager
from jobClassification.components.model_evaluation import ModelEvaluation

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()