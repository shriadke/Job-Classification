from jobClassification.logging import logger
from jobClassification.config.configuration import ConfigurationManager
from jobClassification.components.model_trainer import ModelTrainer

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()