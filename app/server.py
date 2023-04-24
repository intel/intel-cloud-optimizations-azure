import pandas as pd
import uvicorn

from fastapi import FastAPI, File, Form, UploadFile
from loan_app.data import process_data, synthetic_datagen
from loan_app.model import Model
from loan_app.predict import pred
from typing import Optional
from utils.logger import log

app = FastAPI()

@app.get("/ping")
async def ping():

    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"Message":"Server is Running"}

@app.post("/data_processing")
async def data_processing( 
    az_file_path: str = Form(...),
    data_directory: str = Form(...), 
    file: UploadFile = File(...),
    size: Optional[int] = Form(None),
):
    
    """Preprocess Credit Risk data
    This endpoint preprocesses data and stores in data lake or in other structured format. 
    In this codebase, it also handles the expansion of the dataset for benchmarking purposes.
    
    Parameters
    ----------
    az_file_path : str
        volume mount path to azure file share for object storage
    data_directory : str
        name of folder where processed data should be saved in azure file share
    file : file object
        raw csv file for data processing
    size : int, optional
        desired size of final dataset

    Returns
    -------
    API response
        response from server on data processing endpoint
    """
    contents = await file.read()
    
    augmented_data = synthetic_datagen(
        csv_file=contents,
        size=size)
    
    process_data(
        data=augmented_data, 
        az_file_path=az_file_path, 
        data_directory=data_directory,
        size=size)

    return {"Message": f"{file.filename} successfully processed and saved"}

@app.post("/train")
async def train(
    az_file_path: str = Form(...),
    data_directory: str = Form(...), 
    model_directory: str = Form(...),
    model_name: str = Form(...),
    continue_training: Optional[bool] = Form(None),
    size: Optional[int] = Form(None),
):
    """Train the model.
    This endpoint trains the model using the provided path for the data.

    Parameters
    ----------
    az_file_path : str
        volume mount path to azure file share for object storage
    data_directory : str
        name of folder in azure file share where processed data is stored 
    model_directory : str
        name of folder in azure file share for model storage
    model_name : str
        name of model object
    continue_training : bool, optional
        xgboost model parameter for training continuation
    size : int, optional
        size of processed dataset
    
    Returns
    -------
    API response
        response from server on train endpoint
    """

    clf = Model(
        az_file_path=az_file_path, 
        data_directory=data_directory,
        model_directory=model_directory,
        model_name=model_name,
        continue_training=continue_training,
        size=size)
    
    log.info("Loading data")
    clf.load_data()
    log.info("Training model")
    clf.train()
    log.info(f"Saving {model_name} model")
    clf.save()
    scores = clf.validate()
    log.info(f"Validation Scores: {scores}")
    return {"Message": f"{model_name} model succesfully trained and saved", 
            "Validation Scores": scores}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    az_file_path: str = Form(...),
    data_directory: str = Form(...), 
    model_directory: str = Form(...),
    model_name: str = Form(...),
    sample_directory: str = Form(...)
):
    """Prediction endpoint

    Parameters
    ----------
    file : file object
        sample csv file for model inference
    az_file_path : str
        volume mount path to azure file share for object storage
    data_directory : str
        directory in azure file share where data preprocessor is stored
    model_directory : str
        azure file share directory for model storage
    model_name : str
        name of model object
    sample_directory : str
        directory in azure file share to save prediction results

    Returns
    -------
    API response
        response from server on predict endpoint
    """

    sample = await file.read()

    predictions = pred(
        data=sample, 
        az_file_path=az_file_path,  
        data_directory=data_directory,
        model_directory=model_directory,
        model_name=model_name,
        sample_directory=sample_directory)
    
    log.info(f'Prediction Output: {predictions}')
    return {"Message": "Model Inference Complete", "Prediction Output": predictions} 

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, log_level="info")