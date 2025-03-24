from F1_Stint_Prediction.config.configuration import ConfigurationManager
from F1_Stint_Prediction.components.data_transformation import DataTransformation
from F1_Stint_Prediction import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.get_transformed_data()