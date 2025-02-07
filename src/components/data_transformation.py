# All the transformation code here
# categorical to numerical
import sys
import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  #  for complete all the task at once
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer       # for the missing value
from sklearn.pipeline import Pipeline          # for create the pipeline

'''
categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
numerical_columns = ['math_score', 'reading_score', 'writing_score']

'''

class DataTransformationConfig:
    preprocessor

         
