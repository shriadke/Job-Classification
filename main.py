from jobClassification.logging import logger
logger.info("This Project uses custom logger.\n")
logger.info("#################################\n")

from jobClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from jobClassification.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from jobClassification.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from jobClassification.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from jobClassification.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"######## Stage {STAGE_NAME} started. ########")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"######## Stage {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"######## Stage {STAGE_NAME} started. ########")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"######## Stage {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"######## Stage {STAGE_NAME} started. ########")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f"######## Stage {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer stage"
try: 
    logger.info(f"######## Stage {STAGE_NAME} started. ########")   
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f"######## Stage {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation stage"
try: 
    logger.info(f"######## Stage {STAGE_NAME} started. ########")   
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f"######## Stage {STAGE_NAME} completed. ########\n\n################################################")
except Exception as e:
        logger.exception(e)
        raise e