import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    
    def __init__(self):
        pass

    logging.info("Prediction Pipeline Has Started")

    def predict(self, features):
        logging.info("Predict Function Has Started")

        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            logging.info("Succesfully model loaded")

            data_scaled = preprocessor.transform(features)
            #data_scaled = data_scaled.reshape(1,-1)
            logging.info(f"Data Scaled: {data_scaled}")

            pred = model.predict(data_scaled)
            logging.info("Prediction Pipeline Class Ended")
            return pred
 
        except Exception as e:
            logging.info("Exception Occured In Prediction Pipeline")
            raise CustomException(e, sys)
        
class CustomData:

    def __init__(self, 
                 acc_max:float,
                 gyro_max:float,
                 acc_kurtosis:float,
                 gyro_kurtosis:float,
                 #label:str,
                 lin_max:float,
                 acc_skewness:float,
                 gyro_skewness:float,
                 post_gyro_max:float,
                 post_lin_max:float):
        
        self.acc_max = acc_max
        self.gyro_max = gyro_max
        self.acc_kurtosis = acc_kurtosis
        self.gyro_kurtosis = gyro_kurtosis
        self.lin_max = lin_max
        self.acc_skewness = acc_skewness
        self.gyro_skewness = gyro_skewness
        self.post_gyro_max = post_gyro_max
        self.post_lin_max = post_lin_max

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'acc_max':[self.acc_max],
                'gyro_max':[self.gyro_max],
                'acc_kurtosis':[self.acc_kurtosis],
                'gyro_kurtosis':[self.gyro_kurtosis],
                #'label':[self.label],
                'lin_max':[self.lin_max],
                'acc_skewness':[self.acc_skewness],
                'gyro_skewness':[self.gyro_skewness],
                'post_gyro_max':[self.post_gyro_max],
                'post_lin_max':[self.post_lin_max],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info("Exception Occured In Prediction Pipeline")
            raise CustomException(e, sys)
