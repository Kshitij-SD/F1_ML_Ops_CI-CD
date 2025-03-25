import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,precision_score,accuracy_score,f1_score,recall_score
from urllib.parse import urlparse
import numpy as np
import joblib
from F1_Stint_Prediction.utils.common import read_yaml, create_directories, save_json
from F1_Stint_Prediction.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_data(self):
        stint_count_model = joblib.load(self.config.model_path + "/stint_count.joblib")
        X_test_stint_count = pd.read_csv(self.config.test_data_path + "/X_test_stint_count_0.csv")
        y_test_stint_count = pd.read_csv(self.config.test_data_path + "/y_test_stint_count_0.csv")
        
        predicted_stint_counts = stint_count_model.predict(X_test_stint_count)
        
        (rmse, mae, r2) = self.eval_metrics_reg(y_test_stint_count, predicted_stint_counts)
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.root_dir + "/stint_count.json"), data=scores)
    
        for i in range(1,5):
            model = joblib.load(self.config.model_path + f"/Compound_Stint_{i}.joblib")
            X_test = pd.read_csv(self.config.test_data_path + f"/X_test_compound_{i}.csv")
            y_test = pd.read_csv(self.config.test_data_path + f"/y_test_compound_{i}.csv")
            predicted = model.predict(X_test)
            (accuracy,precision,recall,f1) = self.eval_metrics_class(y_test,predicted)
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall,"f1": f1}
            save_json(path=Path(self.config.root_dir + f"/compound_{i}.json"), data=scores)
            
        for i in range(1,5):
            model = joblib.load(self.config.model_path + f"/Stint_len_{i}.joblib")
            X_test = pd.read_csv(self.config.test_data_path + f"/X_test_stint_len_{i}.csv")
            y_test = pd.read_csv(self.config.test_data_path + f"/y_test_stint_len_{i}.csv")
            predicted = model.predict(X_test)
            (rmse, mae, r2) = self.eval_metrics_reg(y_test,predicted)
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.root_dir + f"/stint_len_{i}.json"), data=scores)
            
            
    def eval_metrics_reg(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def eval_metrics_class(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted', zero_division=0)
        recall = recall_score(actual, pred, average='weighted', zero_division=0)
        f1 = f1_score(actual, pred, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1