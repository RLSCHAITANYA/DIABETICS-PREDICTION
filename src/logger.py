#used for logging any execution that happens should be able to log all those information ,the execution everything
# every time we execute any file in this code it will create a log entry and helps us in checking where the error is occuring 
import logging
import os
from datetime import datetime

LOG_FILE=F"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)