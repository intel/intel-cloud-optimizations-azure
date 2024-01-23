# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

import os
import pandas as pd
import daal4py as d4p
import joblib

from io import StringIO
from utils.storage import AZStore
from sklearn.pipeline import Pipeline
from utils.logger import log

def pred(data: object,             
         az_file_path: str, 
         data_directory: str,
         model_directory: str, 
         model_name: str,
         sample_directory: str):
    
    """Returns predictions based on the input `data` using the stored model.

    Parameters
    ----------
    data : object
        sample csv file that was uploaded to the prediction endpoint in server.py
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
    predictions : pd.DataFrame
        predicted probabilities [0,1] where 1 indicates loan default
    """

    data = pd.read_csv(StringIO(str(data, 'utf-8')), encoding='utf-8')

    # loading preprocessor
    az_directory = os.path.join(az_file_path, data_directory)
    with open(os.path.join(az_directory, "preprocessor.sav"), "rb") as file:
        preprocessor = joblib.load(file)
    log.info(f'Successfully loaded data preprocessor from Azure file share '\
                f'in the {data_directory} directory.')
    
    # preprocessor transformations
    preprocess = Pipeline(steps=[("preprocessor", preprocessor)])
    data = preprocess.transform(data)
    data = pd.DataFrame(data)

    # loading model
    model_store = AZStore(
        az_file_path=az_file_path, 
        model_directory=model_directory,
        model_name=model_name)
    d4p_model = model_store.load_model()

    # optimized model inference
    log.info("Starting daal4py inference")
    d4p_probabilities = d4p_model.predict_proba(data)[:,1]
    log.info("Inference Complete")
    
    log.info("Exporting Predictions")
    predictions = pd.DataFrame(columns=["Probability","Prediction"])
    for i, probability in enumerate(d4p_probabilities):
        predictions.loc[i,"Probability"] = probability
        predictions.loc[i,"Prediction"] = "True" if probability > 0.5 else "False"
    
    # save predictions to azure file share
    az_directory = os.path.join(az_file_path, sample_directory)
    if not os.path.exists(az_directory):
        os.makedirs(az_directory)
    predictions.to_csv(os.path.join(az_directory, "sample_predictions.csv"))
    return predictions