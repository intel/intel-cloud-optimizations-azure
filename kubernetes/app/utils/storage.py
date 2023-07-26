import os
import joblib
import daal4py as d4p
from typing import Any
from utils.logger import log

class AZStore:
    
    def __init__(self, 
        az_file_path: str,
        model_directory: str,
        model_name: str):

        """Azure storage functions to save and retrieve model objects
        
        Parameters
        ----------
        az_file_path : str
            volume mount path to azure file share
        model_directory : str
            azure file share directory for model storage
        model_name : str
            name of model object
        """
        self.az_file_path = az_file_path
        self.model_directory = model_directory
        self.az_directory = os.path.join(self.az_file_path, self.model_directory)
        self.model_name = model_name + '.joblib'  

    def load_model(self) -> Any:
        
        with open(os.path.join(self.az_directory, self.model_name), "rb") as clf:
            model = joblib.load(clf)
        
        log.info(f'Successfully loaded model from Azure file share ' \
                 f'in the {self.model_directory} directory.')
        
        log.info(f'Converting XGBoost model to Daal4py.')
        daal_model = d4p.get_gbt_model_from_xgboost(model)
        return daal_model
    
    def save_model(self, model: Any) -> None:
        
        if not os.path.exists(self.az_directory):
            os.makedirs(self.az_directory)
        
        with open(os.path.join(self.az_directory, self.model_name), "wb") as file:
            joblib.dump(model, file)  
        
        log.info(f'Successfully saved {self.model_name} model to Azure file share ' \
                 f'in the {self.model_directory} directory.')

__all__ = ["store"]