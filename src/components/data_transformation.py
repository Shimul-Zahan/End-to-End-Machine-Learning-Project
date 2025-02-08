# All the transformation code here
# categorical to numerical
import sys
import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_file

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  #  for complete all the task at once
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer       # for the missing value
from sklearn.pipeline import Pipeline          # for create the pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_pickle_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def preprocessing_object(self):
        ''' 
            This function is responsibe for all the data transformation
        '''
        try:
            categorical_columns = ['gender','race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_columns = ['reading_score', 'writing_score']
            
            # Create a pipeline for the numerical features or columns
            numerical_pipeline = Pipeline(
                # steps:
                # 1. handle all the missing values
                # 2. Standscaler all the value for get a value in a certain scaling 
                steps=[
                    # (name, what it should be done here)
                    ("fill_missing_imputer", SimpleImputer(strategy='median')),  # for many outlier data er use median for handle the missing value here
                    ("scaler", StandardScaler()),                   # for scaling the data
                ]
            )
            
            categorical_pipeline = Pipeline(
                # steps:
                # 1. handle all the missing values
                # 2. The conver categorical to numerical 
                steps=[
                    # (name, what it should be done here)
                    ('fill_missing_imputer', SimpleImputer(strategy='most_frequent')),
                    ('categorical_to_numerical', OneHotEncoder()),
                    # ('scaler', StandardScaler()),
                ]
            )
            
            logging.info("Numerical columns adds to standscaling and missing value")
            logging.info("Categorcal columns adds to OneHotEncoding and missing value")
            
            # here is the columns transformer for work all the task together
            transformer = ColumnTransformer(
                [
                    # ("definition of the pipeline", what is the pipeline, columns or where to perform pipeline),
                    ("numerical_features_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_features_pipeline", categorical_pipeline, categorical_columns),
                ]
            )
            
            logging.info("Create a transformer and return it for transform the data")
            
            return transformer
        
        except Exception as ex:
            raise CustomException(ex, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
            try:
                train_data = pd.read_csv(train_data_path)
                test_data = pd.read_csv(test_data_path)
                logging.info("Read Train and test data complete")
                logging.info("Obtaining transformer object")
                
                # Get the transformer for transformation working
                transformer = self.preprocessing_object()
                target_columns = 'math_score'
                
                X_train = train_data.drop(columns=[target_columns], axis=1)
                X_test = test_data.drop(columns=[target_columns], axis=1)
                y_train =  train_data[target_columns]
                y_test = test_data[target_columns]
                
                logging.info(f"Applying preprocessing object on train and test data")
                
                X_train_scaled = transformer.fit_transform(X_train)
                X_test_scaled = transformer.transform(X_test)
                
                logging.info("Data transformation completed successfully")
                
                # returning variable here
                train_array = np.c_[X_train_scaled, np.array(y_train).reshape(-1, 1)]
                test_array = np.c_[X_test_scaled, np.array(y_test).reshape(-1, 1)]
                
                logging.info("Saved preprocessing object")
                
                save_file(
                    file_path = self.data_transformation_config.preprocessor_pickle_file_path,
                    obj = transformer
                )
                
                return(
                    train_array,
                    test_array,
                    self.data_transformation_config.preprocessor_pickle_file_path
                )
                
            except Exception as ex:
                raise CustomException(ex, sys)
         
