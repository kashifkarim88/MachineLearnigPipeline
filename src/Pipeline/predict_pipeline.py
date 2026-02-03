import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    # Singleton pattern: model & preprocessor loaded once
    _model = None
    _preprocessor = None

    def __init__(self):
        try:
            if PredictPipeline._model is None or PredictPipeline._preprocessor is None:
                # Get the absolute path to the artifacts folder
                base_dir = os.path.dirname(os.path.abspath(__file__))  # location of this file
                preprocessor_path = os.path.join(base_dir, "..", "..", "artifacts", "preprocessor.pkl")
                model_path = os.path.join(base_dir, "..", "..", "artifacts", "model.pkl")

                print(f"Loading preprocessor from: {preprocessor_path}")
                print(f"Loading model from: {model_path}")

                PredictPipeline._preprocessor = load_object(preprocessor_path)
                PredictPipeline._model = load_object(model_path)
                print("Model and preprocessor loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            print("Transforming features...")
            data_scaled = PredictPipeline._preprocessor.transform(features)
            print("Making predictions...")
            preds = PredictPipeline._model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 gender: str, 
                 race_ethnicity: str, 
                 parental_level_of_education: str, 
                 lunch: str, 
                 test_preparation_course: str, 
                 reading_score: int, 
                 writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
