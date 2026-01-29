import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            if model_name == "CatBoosting Regressor":
                model.fit(X_train, y_train)
            else:
                gs = GridSearchCV(
                    model,
                    param[model_name],
                    cv=3,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score
            trained_models[model_name] = model

            logging.info(f"{model_name} R2 score: {test_score}")

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
