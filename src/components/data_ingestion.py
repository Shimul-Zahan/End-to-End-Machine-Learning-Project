''' 
    # data ingestion means reading the data from various sources
    TODO: 
    ### Import all the dependencies here
    1. Read the data
    2. Split the data
    3. 
'''

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# read datasets from local storage
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestions method or components")
        try:
            dataset = pd.read_csv('notebook/data/stud.csv')
            logging.info(f"shape of datasets here is {dataset.shape}")
            logging.info("read the dataset ad dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            dataset.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split here")
            train_set, test_set= train_test_split(dataset, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of data is complete")
            
            # get all the categorical and numerical features 
            categorical_columns = [column for column in dataset.columns if dataset[column].dtype == 'object']
            numerical_columns = [column for column in dataset.columns if dataset[column].dtype != 'object']
            
            print(dataset['math_score'])
        
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as ex:
            raise CustomException(ex, sys)
        

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()