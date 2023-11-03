import sys
import os
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.simplefilter("ignore")

from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformConfig:
    preprocess_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformConfig

    def get_transformation_object(self):

        try:
            logging.info("Data Transform Initiated")

            ## Separating Numerical Features & Categorical Features

            numerical_features = ["acc_max", "gyro_max", "acc_kurtosis", "gyro_kurtosis",
                      "lin_max", "acc_skewness", "gyro_skewness", "post_gyro_max",
                      "post_lin_max"]
            
            # Scaling
            numeric_transformer = StandardScaler()

            preprocessor = ColumnTransformer(
                transformers = [
                    ("num", numeric_transformer, numerical_features)
                    ]
                    )
            logging.info('pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info("Error Occured In Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading training set and testing set is started")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Dropping Unneccasary Columns")

            ## Dropping Unneccasary Columns
            train_data = train_data.drop(["Unnamed: 0", "label"],axis=1)
            test_data = test_data.drop(["Unnamed: 0", "label"], axis=1)

            logging.info("Unneccasary columns successfully dropped")

            logging.info("Samples of training set and testing set")

            logging.info(f'Train Dataframe Head: \n{train_data.head().to_string()}')
            logging.info(f"Test Dataframe Head: \n{test_data.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_transformation_object()

            target_column_name = "fall"
            drop_columns = target_column_name

            input_feature_train_df = train_data.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df=test_data.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_data[target_column_name]

            ## Transforming using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # We are just concatenating our input train and output train
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # we are just concatenating our input test and output test
            # above 2 lines edhukku use panni irukom na, data transformation panna aprm neraya rows and columns vara chance irukku
            # data va input ah kudukurappa, again modhala irukku run aaga vida mudiyadhu, so just concatenating with array
            # idhu pandrapa it will run very quickly.
            
            save_object(

                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path,
            )


        except Exception as e:
            logging.inf("Error occured in initiate data transformation function")
            raise CustomException(e, sys)
