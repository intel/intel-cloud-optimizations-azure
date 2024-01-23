# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

import os
import daal4py as d4p
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from utils.logger import log
from utils.storage import AZStore

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import classification_report, roc_auc_score

np.random.seed(42)

class Model():
        
    def __init__(self, 
            az_file_path: str, 
            data_directory: str,
            model_directory: str,
            model_name: str, 
            continue_training: bool = False,
            size: int = 4000000):
        
        """Class to manage the training and validation of XGBoost and daal4py models.

        Parameters
        ----------
        az_file_path : str
            volume mount path to azure file share for object storage
        data_directory : str
            directory in azure file share where processed data is stored 
        model_directory : str
            azure file share directory for model storage
        model_name : str
            name of model object
        continue_training : bool, optional
            xgboost model parameter for training continuation, by default False
        size : int, optional
            size of processed dataset, by default 4M rows
        """

        self.az_file_path = az_file_path
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.model_name = model_name
        self.continue_training = continue_training     
        self.size = size   
        self.store = AZStore(
            az_file_path = self.az_file_path,
            model_directory = self.model_directory,
            model_name = self.model_name
        )
        self.X_train = []
        self.y_train = []
        self.y_test = []
        self.X_test = []
        self.dtrain = []
    
    def load_data(self):
        
        # load data
        file_path = os.path.join(self.az_file_path, self.data_directory, f"X_train_{self.size}.csv")
        self.X_train = pd.read_csv(file_path, header=None)

        file_path = os.path.join(self.az_file_path, self.data_directory, f"y_train_{self.size}.csv")
        self.y_train = pd.read_csv(file_path, header=None)

        file_path = os.path.join(self.az_file_path, self.data_directory, f"X_test_{self.size}.csv")
        self.X_test = pd.read_csv(file_path, header=None)

        file_path = os.path.join(self.az_file_path, self.data_directory, f"y_test_{self.size}.csv")
        self.y_test = pd.read_csv(file_path, header=None)

        log.info('Successfully loaded data from Azure file share '\
                 f'in the {self.data_directory} directory.')
        
        self.dtrain = xgb.DMatrix(self.X_train.values, self.y_train.values)

    def train(self):

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "nthread": 4,  # num_cpu
            "tree_method": "hist",
            "learning_rate": 0.02,
            "max_depth": 10,
            "min_child_weight": 6,
            "n_jobs": 4,  # num_cpu,
            "verbosity": 1
        }

        if self.continue_training == False or self.continue_training == None:
            log.info(f"Training initial {self.model_name} model")
            self.clf = xgb.train(params=params, 
                                 dtrain=self.dtrain, 
                                 num_boost_round=500)
        else:
            log.info(f"Loading {self.model_name} model from {self.model_directory} directory")
            model_path = os.path.join(self.az_file_path, self.model_directory, self.model_name+'.joblib')
            try: 
                model = joblib.load(model_path)
                log.info(f"Continuing {self.model_name} training")
                self.clf  = xgb.train(
                    params=params, dtrain=self.dtrain, xgb_model=model, num_boost_round=500)
            except:
                print(f"{self.model_name} model not found")

    def validate(self):
        """Function to compute classification metrics for model validation.

        Returns
        -------
        results : dict
             A dictionary containing the following keys:
                'precision' (dict): A dictionary of precision scores, with keys 'Non-Default' and 'Default'.
                'recall' (dict): A dictionary of recall scores, with keys 'Non-Default' and 'Default'.
                'f1-score' (dict): A dictionary of F1 scores, with keys 'Non-Default' and 'Default'.
                'support' (dict): A dictionary of support values, with keys 'Non-Default' and 'Default'.
                'auc' (float): The area under the receiver operating characteristic curve (AUC).
        """
        d4p_model = self.store.load_model()

        y_hat = d4p_model.predict_proba(X_test)[:,1]
        
        auc = roc_auc_score(self.y_test, y_hat)
        results = classification_report(
            self.y_test,
            y_hat > 0.5,
            target_names=["Non-Default", "Default"],
            output_dict=True
        )
        #results.update({"AUC": auc})

        # save model results 
        az_directory = os.path.join(self.az_file_path, self.model_directory)
        if not os.path.exists(az_directory):
            os.makedirs(az_directory)

        results = pd.DataFrame(results).transpose()
        results.to_csv(os.path.join(az_directory, f"{self.model_name}_results.csv"))
        return results

    def save(self):
        log.info("Saving model")
        self.store.save_model(model = self.clf)