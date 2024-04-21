import sys # for exception handling
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # to load our pickle files


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # this function tells what our model is predicting.

            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl' # preprocessor_path is responsible feature engineering & feature selection.
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            # Now, once the data is loaded, we scale the data.

            data_scaled = preprocessor.transform(features)

            # Now, model will do the prediction.

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)




class CustomData:  
#CustomData is responsible in mapping all the inputs that we 
# are giving in html to backend w.r.t particular values.
    def __init__(
            self,
            gender:str,
            race_ethnicity: str,
            parental_level_of_education,
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
        # this function return all the inputs in the form of dataframe.
        # As we train our model in the form of dataframe
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        