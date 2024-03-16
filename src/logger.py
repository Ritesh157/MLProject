import logging
import os
from datetime import datetime

#Creating log file

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# we need to give the path of log file
# "logs" means every file with logs

logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

#exist_ok=True means if the file exist, then keep appending that file.
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH =os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)

