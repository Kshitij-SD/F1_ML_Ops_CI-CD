import pandas as pd
import xgboost as xgb
import joblib
import os
from F1_Stint_Prediction.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_data(self):
        # Load Stint Count (Regression)
        X_train_stint_count = pd.read_csv(self.config.train_data_path + "/X_train_stint_count_0.csv")
        y_train_stint_count = pd.read_csv(self.config.train_data_path + "/y_train_stint_count_0.csv")

        X_test_stint_count = pd.read_csv(self.config.train_data_path + "/X_test_stint_count_0.csv")
        y_test_stint_count = pd.read_csv(self.config.train_data_path + "/y_test_stint_count_0.csv")

        self.train_Regressor(X_train_stint_count, y_train_stint_count,X_test_stint_count,y_test_stint_count ,"stint_count")

        # Compound Classification (per stint)
        for i in range(1, 5):
            X_train = pd.read_csv(self.config.train_data_path + f"/X_train_compound_{i}.csv")
            y_train = pd.read_csv(self.config.train_data_path + f"/y_train_compound_{i}.csv")

            X_test = pd.read_csv(self.config.train_data_path + f"/X_test_compound_{i}.csv")
            y_test = pd.read_csv(self.config.train_data_path + f"/y_test_compound_{i}.csv")

            self.train_Classifier(X_train, y_train,X_test,y_test, f"Compound_Stint_{i}")
            # You can now use X_test and y_test if you want to evaluate

        # Stint Length Regression (per stint)
        for i in range(1, 5):
            X_train = pd.read_csv(self.config.train_data_path + f"/X_train_stint_len_{i}.csv")
            y_train = pd.read_csv(self.config.train_data_path + f"/y_train_stint_len_{i}.csv")

            X_test = pd.read_csv(self.config.train_data_path + f"/X_test_stint_len_{i}.csv")
            y_test = pd.read_csv(self.config.train_data_path + f"/y_test_stint_len_{i}.csv")

            self.train_Regressor(X_train, y_train,X_test,y_test, f"Stint_len_{i}")
            # Use X_test and y_test as needed!


    def train_Classifier(self,X_train,y_train,X_test,y_test, name):
        model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            early_stopping_rounds=self.config.early_stopping_rounds,
            learning_rate=self.config.learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)])

        joblib.dump(model, os.path.join(self.config.root_dir, f"{name}.joblib"))
    
    def train_Regressor(self,X_train,y_train,X_test,y_test, name):
        model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            early_stopping_rounds=self.config.early_stopping_rounds,
            learning_rate=self.config.learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)])

        joblib.dump(model, os.path.join(self.config.root_dir, f"{name}.joblib"))