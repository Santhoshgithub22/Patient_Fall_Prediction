import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):

        try:
            logging.info("Splitting Dependent And Independent Variable From Training Data And Test Data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info(f'X_Train Dataframe Head: \n{X_train}')
            logging.info(f'y_Train Dataframe Head: \n{y_train}')
            logging.info(f'X_Test Dataframe Head: \n{X_test}')
            logging.info(f'y_Train Dataframe Head: \n{X_test}')

            ## Converting this array into dataframe

            y_test = y_test.reshape(-1,1)

            #X_train = pd.DataFrame(X_train)
            #y_train = pd.DataFrame(y_train)
            #X_test = pd.DataFrame(X_test)
            #y_test = pd.DataFrame(y_test)

            logging.info("Successfully Independent And Dependent Splitted For Training Set And Test Set")

            logging.info("Ready To Do Model")

            models ={
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Classifier" : SVC(),
                "Naive Bayes": GaussianNB(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Ada Boosting": AdaBoostClassifier(),
                "K-Neighbours" : KNeighborsClassifier()
                }
            logging.info("Succesfully model was fitted")

            logging.info("Model Evaluation Has Started")

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')


            logging.info("Model Evaluation Has Completed")

            ## To get the best model score from dictionary

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )


        except Exception as e:
            logging.info("Error Occured At Model Training")
            raise CustomException(e, sys)