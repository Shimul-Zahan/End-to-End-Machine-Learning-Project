import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.exception import CustomException

def save_file(file_path, obj):
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file:
            dill.dump(obj, file)
    
    except Exception as ex:
        raise CustomException(ex, sys)