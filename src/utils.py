import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_file(file_path, obj):
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    
    except Exception as ex:
        raise CustomException(ex, sys)
    
def evalute_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for model_name, model in models.items():
            # print(f"{model_name}: {model}")
            param_grid = params.get(model_name, {})
            print(param_grid)
            
            if param_grid:  # If hyperparameters exist for this model, use GridSearchCV
                grid_search = GridSearchCV(model, param_grid, cv=3)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"{model_name}: {best_params}")
            else:
                best_model = model  # If no hyperparameters provided, use default model
                best_model.fit(X_train, y_train)

            
            # here we predict/evalue this model
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # evalute or measure the score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            print(f"{model_name} train score: {train_model_score}")
            print(f"{model_name} test score: {test_model_score}")
            
            # return this result
            # Store results
            report[model_name] = test_model_score
            
        return report
    
    except Exception as ex:
        raise CustomException(ex, sys)
    

def load_file(file_path):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file)
    
    except Exception as ex:
        raise CustomException(ex, sys)
    
    
    
''' 
X_train | y_train
------------------
X_test  | y_test
'''