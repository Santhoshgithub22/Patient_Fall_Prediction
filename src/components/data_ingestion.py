import os
import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation

import pandas as pd
from dataclasses import dataclass

# Initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")

# Creating a class for data ingestion

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Started")

        try:
            train_data = pd.read_csv(os.path.join("notebooks/data/Train.csv"))
            test_data = pd.read_csv(os.path.join("notebooks/data/Test.csv"))

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            #train_data.to_csv(self.ingestion_config.raw_data_path, index=False)
            #test_data.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(f"Train Dataframe Sample's: \n{train_data.sample(2).to_string()}")
            logging.info(f"Test Dataframe Sample's: \n{test_data.sample(2).to_string()}")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Part Successfully Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured in Data Ingestion stage")
            raise CustomException(e, sys)

        
## Run Data Ingestion

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)