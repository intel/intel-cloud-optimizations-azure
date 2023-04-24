# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

import os
import numpy as np
import pandas as pd
import joblib

from io import StringIO
from utils.logger import log

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer


def synthetic_datagen(
        csv_file: object,
        size: int = 4000000):
    
    """Generates additional synthetic data for benchmarking and testing purposes. 

    Parameters
    ----------
    csv_file : object
        csv file that was uploaded to the data_processing endpoint in server.py
    size : int, optional
        desired size of final dataset, default 4M rows
    
    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe with the original data or original plus synthetic augmentation.
    """

    data = pd.read_csv(StringIO(str(csv_file, 'utf-8')), encoding='utf-8')

    # number of rows to generate
    if size < data.shape[0]:
        pass
    else:
        log.info(f"Generating {size:,} rows of data")
        repeats = size // len(data)
        data = data.loc[np.repeat(data.index.values, repeats + 1)]
        data = data.iloc[:size]
        
        # perturbing all int/float columns
        person_age = data["person_age"].values + np.random.randint(
            -1, 1, size=len(data)
        )
        person_income = data["person_income"].values + np.random.normal(
            0, 10, size=len(data)
        )
        person_emp_length = data[
            "person_emp_length"
        ].values + np.random.randint(-1, 1, size=len(data))
        loan_amnt = data["loan_amnt"].values + np.random.normal(
            0, 5, size=len(data)
        )
        loan_int_rate = data["loan_int_rate"].values + np.random.normal(
            0, 0.2, size=len(data)
        )
        loan_percent_income = data["loan_percent_income"].values + (
            np.random.randint(0, 100, size=len(data)) / 1000
        )
        cb_person_cred_hist_length = data[
            "cb_person_cred_hist_length"
        ].values + np.random.randint(0, 2, size=len(data))
        
        # perturbing all binary columns
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["person_home_ownership"].unique(), len(data)
        )
        person_home_ownership = np.where(
            perturb_idx, data["person_home_ownership"], random_values
        )
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["loan_intent"].unique(), len(data)
        )
        loan_intent = np.where(perturb_idx, data["loan_intent"], random_values)
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["loan_grade"].unique(), len(data)
        )
        loan_grade = np.where(perturb_idx, data["loan_grade"], random_values)
        perturb_idx = np.random.rand(len(data)) > 0.1
        random_values = np.random.choice(
            data["cb_person_default_on_file"].unique(), len(data)
        )
        cb_person_default_on_file = np.where(
            perturb_idx, data["cb_person_default_on_file"], random_values
        )
        data = pd.DataFrame(
            list(
                zip(
                    person_age,
                    person_income,
                    person_home_ownership,
                    person_emp_length,
                    loan_intent,
                    loan_grade,
                    loan_amnt,
                    loan_int_rate,
                    data["loan_status"].values,
                    loan_percent_income,
                    cb_person_default_on_file,
                    cb_person_cred_hist_length,
                )
            ),
            columns=data.columns,
        )

        augmented_data = data.drop_duplicates()
        assert len(augmented_data) == size
        augmented_data.reset_index(drop=True)

        return augmented_data

    return data

def process_data(
        data: pd.DataFrame, 
        az_file_path: str,
        data_directory: str,
        size: int = 4000000):
    
    """Utility function for preprocessing loan default data.

    Parameters
    ----------
    data : pd.DataFrame
        data that has been augmented or loaded by the synthetic_datagen function
    az_file_path : str
        volume mount path to azure file share for object storage
    data_directory : str
        directory in azure file share where processed data should be saved
    size : int, optional
        size of final dataset, default 4M rows
    """
    
    # train test split and apply data transformations
    log.info("Creating training and test sets")
    train, test = train_test_split(data, test_size=0.25, random_state=0)
    num_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    pow_transformer = PowerTransformer()
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                num_imputer,
                [
                    "loan_int_rate",
                    "person_emp_length",
                    "cb_person_cred_hist_length",
                ],
            ),
            (
                "pow",
                pow_transformer,
                ["person_age", "person_income", "loan_amnt", "loan_percent_income"],
            ),
            (
                "cat",
                cat_transformer,
                [
                    "person_home_ownership",
                    "loan_intent",
                    "loan_grade",
                    "cb_person_default_on_file",
                ],
            ),
        ],
        remainder="passthrough",
    )

    # data processing pipeline
    preprocess = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train = train.drop(["loan_status"], axis=1)
    y_train = train["loan_status"]
    X_train = preprocess.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    log.info(f"preprocess named steps: {preprocess.named_steps['preprocessor']}")
    X_test = test.drop(["loan_status"], axis=1)
    y_test = test["loan_status"]
    X_test = preprocess.transform(X_test)
    X_test = pd.DataFrame(X_test)
    
    # create directory in azure file share to save processed data    
    az_directory = os.path.join(az_file_path, data_directory)
    if not os.path.exists(az_directory):
        os.makedirs(az_directory)

    # export processing pipeline for inference data processing
    with open(os.path.join(az_directory, "preprocessor.sav"), "wb") as file:
        joblib.dump(preprocessor, file)   

    # save processed data to azure file share
    X_train.to_csv(os.path.join(az_directory, f"X_train_{size}.csv"), 
                   index=False, header=None)
    y_train.to_csv(os.path.join(az_directory, f"y_train_{size}.csv"), 
                   index=False, header=None)
    X_test.to_csv(os.path.join(az_directory, f"X_test_{size}.csv"), 
                  index=False, header=None)
    y_test.to_csv(os.path.join(az_directory, f"y_test_{size}.csv"), 
                  index=False, header=None)
    
    log.info('Successfully saved data and data preprocessor to Azure file share ' \
             f'in the {data_directory} directory.')