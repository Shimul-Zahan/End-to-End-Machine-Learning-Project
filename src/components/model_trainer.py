# All the codes for model trainignand training relates
import os
import sys
from dataclasses import dataclass
import pandas as pd

# All model library
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# my module
from src.exception import CustomException
from src.logger import logging
from src.utils import save_file, evalute_model

# for every file we need a config file
@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')
    
# Responsible for training my model
class ModelTrainer:
    def __init__(self):
        self.train_model_file_path_config = ModelTrainerConfig()
        
    def initiate_model_trainier(self, train_array, test_array):
        try:
            logging.info("Split the train test data")
            # print(pd.DataFrame(train_array), test_array, preprocessor_path)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            # create a dict for model training
            models = {
                # "model name", "model class"
                "RF": RandomForestRegressor(),
                "DT": DecisionTreeRegressor(),
                "GB": GradientBoostingRegressor(),
                "LR": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "ADB": AdaBoostRegressor()
            }
            
            model_report: dict = evalute_model(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
            )
            
            # Identify the best model based on the highest test score
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]
            
            ''' 
            print(f"best model name: {best_model_name}")
            print(f"best model: {best_model}")
            print(f"best model score: {best_model_score}")
            best model name: LR
            best model: LinearRegression()
            best model score: 0.8804332983749565
            '''
            
            save_file(
                file_path=self.train_model_file_path_config.train_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best Model Found: {best_model_name} with score: {best_model_score}")
            
            return best_model_name, best_model_score
        
        except Exception as ex:
            raise CustomException(ex, sys)

