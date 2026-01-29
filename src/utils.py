# Common Functionalaity we have to use in multiple places
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj: The Python object to be saved.

    Raises:
        Exception: If there is an error during the saving process.
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, model):
    """
    Evaluate multiple machine learning models and return their R2 scores.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        Xtest (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        model (dict): A dictionary of model names and their corresponding instantiated objects.

    Returns:
        dict: A dictionary with model names as keys and their R2 scores as values."""
    
    try:
        report = {}
        for i in range(len(model)):
            model_name = list(model.keys())[i]
            model_obj = list(model.values())[i]
            logging.info(f"Training the model: {model_name}")
            model_obj.fit(X_train, y_train)
            y_test_pred = model_obj.predict(X_test)
            y_train_pred = model_obj.predict(X_train)
            test_model_score = r2_score(y_test, y_test_pred)
            train_model_score = r2_score(y_train, y_train_pred)
            report[list(model.keys())[i]] = test_model_score
            logging.info(f"{model_name} model test r2 score: {test_model_score}")
            logging.info(f"{model_name} model train r2 score: {train_model_score}")
        return report
    except Exception as e:
        raise CustomException(e, sys)