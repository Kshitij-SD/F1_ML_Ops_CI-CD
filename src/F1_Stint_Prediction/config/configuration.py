from F1_Stint_Prediction.constants import *
from F1_Stint_Prediction.utils.common import read_yaml, create_directories
from F1_Stint_Prediction.entity.config_entity import DataIngestionConfig
from F1_Stint_Prediction.entity.config_entity import DataTransformationConfig
from F1_Stint_Prediction.entity.config_entity import ModelTrainerConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            cache_dir = config.cache_dir,
            raw_data_dir = config.raw_data_dir,
            data_dir = config.data_dir
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.Xgboost

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            n_estimators= params.n_estimators,
            early_stopping_rounds= params.early_stopping_rounds,
            learning_rate= params.learning_rate
        )

        return model_trainer_config