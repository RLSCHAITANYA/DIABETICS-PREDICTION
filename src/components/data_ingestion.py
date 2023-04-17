# read the data from main data source

import os
import sys

from src.exception import CustomException  #connectiong to the exception catching part we created
from src.logger import logging   #connectiong to logging part
import pandas as pd 



from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_train import ModelTrainer
from src.components.model_train import ModelTrainerConfig


#here we create a class so that when ever we perform a data ingestion there should be some input that may be probably
# required by data ingestion component the input can be like where to save training data test data raw data paths 
# 1.
@dataclass #inside a class to define the class variable we use init but using @dataclass we can directly define variables
class DataIngestionConfig:
    train_data_pata:str=os.path.join('artifacts',"train.csv")
    test_data_pata:str=os.path.join('artifacts',"test.csv")     #inputs we provided
    raw_data_pata:str=os.path.join('artifacts',"data.csv")

# 2.
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\gender.csv')
            logging.info('read the dataset as data frame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_pata),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_pata,index=False,header=True)

            logging.info('train_test_split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_pata,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_pata,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_pata,
                self.ingestion_config.test_data_pata,
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
            
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initate_model_training(train_arr,test_arr))


#1.here we initially create a class DataIngestionConfig where we create a path for train test and raw data 
# and save them in a folder called as artifacts as train.csv,test.csv and raw.csv
# 2.here we create another class DataIngestion where we define a function1--> which initaite the path configurations
# of train_test_raw into ingestion config function2--->where we read the data.we initially read the data adn store it
# into a directory do a train_test_split and save them differntly ito train_set and test_set and we return the sets
# (loggging info :- gives the info till where the execution process has been completed )