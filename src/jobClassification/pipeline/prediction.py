import pandas as pd
import csv
import os
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
import tqdm

from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers.util import cos_sim

from sklearn.linear_model import LogisticRegression

from jobClassification.config.configuration import ConfigurationManager
from jobClassification.logging import logger
from jobClassification.util import *


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.onet_embd_path=config.onet_embeddings #"data/processed/embeddings/onet_custom_model_1/embd.pkl"
        self.model_ckpt = config.model_path # "models/"
        self.id_to_onet_dict, self.onet_to_id_dict = util.get_onet_dicts()
        with open(config.lr_model_path, "rb") as fIn:
            self.LRmodel = pickle.load(fIn)  
    

    

    
    def predict(self,job_title,job_body, top_k):
                
        logger.info(f"Searched for:{job_title,job_body, top_k}")
        logger.info("Getting all ONETs!!")
        
        job_df = util.get_job_embd_df_frm_title_body(job_title, job_body, model_ckpt=self.model_ckpt)

        
        X = job_df["JOB_EMBD"].to_list()

        loaded_model = self.LRmodel

        y_pred = loaded_model.predict(X)
        prob = loaded_model.predict_proba(X)
        
        all_classes = list(range(0,len(self.onet_to_id_dict)))
        all_classes = np.array(sorted(all_classes))

        new_prob = np.zeros((prob.shape[0], all_classes.size))
        # Set the columns corresponding to clf.classes_
        new_prob[:, all_classes.searchsorted(loaded_model.classes_)] = prob
        results = []
        for i in range(len(new_prob)):
            preds_idx = np.argsort(new_prob[i])[::-1][:top_k]
            str_preds = [self.id_to_onet_dict[str(idx)] for idx in preds_idx] 
            print("res:",i, str_preds)
            results.append(str_preds)
        

        output_df = pd.DataFrame({
            "Offer" : results
        })

        return output_df