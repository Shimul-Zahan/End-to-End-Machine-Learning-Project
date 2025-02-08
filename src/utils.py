import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.exception import CustomException

from sklearn.metrics import r2_score

def save_file(file_path, obj):
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    
    except Exception as ex:
        raise CustomException(ex, sys)
    
def evalute_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        for model_name, model in models.items():
            # print(f"{model_name}: {model}")
            model.fit(X_train, y_train)   # here train the model
            
            # here we predict/evalue this model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # evalute or measure the score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # print(f"{model_name} train score: {train_model_score}")
            # print(f"{model_name} test score: {test_model_score}")
            
            # return this result
            report[model_name] = test_model_score
            
        return report
    
    except Exception as ex:
        raise CustomException(ex, sys)
    
    
    
''' 
X_train | y_train
------------------
X_test  | y_test

'''