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
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider
                    'weights': ['uniform', 'distance'],  # Weight function used in prediction
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm to compute nearest neighbors
                    'leaf_size': [10, 20, 30, 40, 50],  # Leaf size for BallTree and KDTree
                    'p': [1, 2]  # Power parameter for Minkowski distance (1=Manhattan, 2=Euclidean)
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }  
            }
            
            model_report: dict = evalute_model(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models,
                params=params,
            )
            
            # Identify the best model based on the highest test score
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]
            
            print(f"{best_model}")
            
            
            save_file(
                file_path=self.train_model_file_path_config.train_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best Model Found: {best_model_name} with score: {best_model_score}")
            
            return best_model_name, best_model_score
        
        except Exception as ex:
            raise CustomException(ex, sys)

