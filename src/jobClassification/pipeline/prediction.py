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
from jobClassification.utils.common import *


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.onet_embd_path=self.config.onet_embeddings #"data/processed/embeddings/onet_custom_model_1/embd.pkl"
        self.model_ckpt = self.config.model_path # "models/"
        self.id_to_onet_dict, self.onet_to_id_dict = get_onet_dicts()
        with open(self.config.lr_model_path, "rb") as fIn:
            self.LRmodel = pickle.load(fIn)  
    
    def compare_docs_and_queries(job_embeddings, onet_embeddings, top_k=1):
        """ 
        This method performs semantic search operation that finds top_k ONET embeddings for the given job embeddings 
        using SBERT util's semantic_search method with cosine similarity. 
        
        Input: job_embeddings: ndarray --> List/array of input job embeddings.
            onet_embeddings: ndarray --> List/array of input ONET embeddings.
            top_k: int --> (Optional) top K results to return

        Output: hits: list[list[{"corpus_id", "score"}]] --> Returns a list with one entry for each query. 
                                Each entry is a list of dictionaries with the keys ‘corpus_id’ (id of ONETs) and ‘score’, 
                                sorted by decreasing cosine similarity scores
        """
        
        hits = util.semantic_search(job_embeddings, onet_embeddings, top_k=top_k)

        return hits

    def process_hits(all_hits,job_df, onet_embd_df):
        """ 
        This method processes the hits returned by semantic search and maps them to corresponding inputs. 
        
        Input: all_hits: ndarray --> List/array of hits.
            job_df: pd.DataFrame --> Input Job Dataframe.
            onet_embd_df: pd.DataFrame --> Input O*NET Dataframe.

        Output: result_df: pd.DataFrame --> Output Dataframe with PRED_ONETS
        """
        result_df = job_df.copy()
        results = []
        
        for id in range(len(all_hits)):
            hits = all_hits[id]
            k_res = []
            for hit in hits:
                pred_hit = onet_embd_df["ONET_NAME"][hit['corpus_id']]
                #print("\t{:.3f}\t{}".format(hit['score'], pred_hit))
                k_res.append(pred_hit)
            results.append(k_res)
        result_df["RANK"] = pd.Series(list(range(1, len(results+1))))
        result_df["O*NET NAMES"] = pd.Series([arr for arr in results])

        return result_df
    
    def predict(self,job_title,job_body, top_k, test_model="LR"):
                
        logger.info(f"Searched for:{job_title,job_body, top_k}")
        logger.info("Getting all ONETs!!")
        
        job_df = get_job_embd_df_frm_title_body(job_title, job_body, model_ckpt=self.model_ckpt)

        if test_model == "LR":
            X = job_df["JOB_EMBD"].to_list()

            loaded_model = self.LRmodel

            y_pred = loaded_model.predict(X)
            prob = loaded_model.predict_proba(X)
            
            all_classes = list(range(0,len(self.onet_to_id_dict)))
            all_classes = np.array(sorted(all_classes))
            print(all_classes.shape)

            new_prob = np.zeros((prob.shape[0], all_classes.size))
            print(new_prob.shape)
            # Set the columns corresponding to clf.classes_
            new_prob[:, all_classes.searchsorted(loaded_model.classes_)] = prob
            results = []
            for i in range(len(new_prob)):
                preds_idx = np.argsort(new_prob[i])[::-1][:top_k]
                str_preds = [(rank+1, self.id_to_onet_dict[str(idx)]) for rank,idx in enumerate(preds_idx)] 
                # print("res:",i, str_preds)
                results.append(str_preds)
            

            output_df = pd.DataFrame(results[0],columns=["Rank","O*NET NAMES"])

        elif test_model == "SBERT":

            onet_embd_df = get_onet_embeddings(onet_embd_path=self.onet_embd_path)

            all_hits = compare_docs_and_queries(np.array(job_df["JOB_EMBD"].to_list()), np.array(onet_embd_df["ONET_EMBD"].to_list()), top_k=top_k)

            output_df = process_hits(all_hits,job_df, onet_embd_df)

        return output_df