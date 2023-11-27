from jobClassification.logging import logger
from jobClassification.config.configuration import ConfigurationManager
from jobClassification.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
