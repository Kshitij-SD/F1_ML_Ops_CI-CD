from F1_Stint_Prediction.config.configuration import ConfigurationManager
from F1_Stint_Prediction import logger
from F1_Stint_Prediction.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.eval_data()