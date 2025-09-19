# This script orchestrates the full machine learning pipeline for image classification.
# It sequentially runs the data ingestion, base model preparation, model training, and evaluation stages.
# Each stage logs its progress and handles exceptions gracefully.

from ImageClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ImageClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from ImageClassification.pipeline.stage_03_training import ModelTrainingPipeline
from ImageClassification.pipeline.stage_04_evaluation import EvaluationPipeline
from ImageClassification import logger

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>> stage {STAGE_NAME} started <<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>> stage {STAGE_NAME} completed <<<\n\nX======X")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare base model"

try:
    logger.info(f"**********************")
    logger.info(f">>> stage {STAGE_NAME} started <<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>> stage {STAGE_NAME} completed <<<\n\nX======X")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 3: Training
STAGE_NAME = "Training"

try:
    logger.info(f"**************************")
    logger.info(f">>>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
