from F1_Stint_Prediction.config.configuration import ConfigurationManager
from F1_Stint_Prediction.components.model_trainer import ModelTrainer
from F1_Stint_Prediction import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.get_data()