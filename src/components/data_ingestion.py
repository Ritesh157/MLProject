import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts', 'train.csv')
    # artifacts is basically a folder to see our output. 
    # All the output will be stored in artifacts folder.
    # train.csv is our file name.

    test_data_path:str=os.path.join('artifacts', 'test.csv') # for test data
    raw_data_path:str=os.path.join('artifacts', 'data.csv') # for raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        # ingestion_config variable consist of above three path.
        # when we call ataIngestionConfig(), then above three path get save in ingestion_config variable 
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as Dataframe')

            # create a folder with the help of train_data_path, test_data_path, raw_data_path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # w.r.t raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # saving our training data in train_data_path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # saving our test data in test_data_path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion() # Initiate Data Ingestion Object
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)