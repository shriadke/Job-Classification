import pandas as pd
import csv
import numpy as np
import pickle
import os
from box.exceptions import BoxValueError
import yaml
from jobClassification.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers.util import cos_sim

from sklearn.linear_model import LogisticRegression

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def clean_text(text, stop_words=stopwords.words('english'), punct=string.punctuation, special_tokens=[]):
        """ 
        This method returns cleaned String from an input string. 
        Removes stop words, punctuations, numbers and any special tokens given.
        
        Input:  text : str                  --> input string to be cleaned
                stop_words : list[str]      --> (Optional) list of stop words to be removed from the text
                punct : list[str]           --> (Optional) list of punctuations to be removed from the text
                special_tokens : list[str]  --> (Optional) list of special words to be removed from the text

        Output: text: string:  cleaned text
        """

        text= text.lower()
        
        text = text.replace("\n"," ")
        
        text = text.replace(r'[0-9]+', ' ')
        text = text.replace(r'[^\w\s]', ' ')
        text = text.replace(r'[^a-zA-Z]', ' ')
        for p in punct:
            text = text.replace(p," ") 
            
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([word for word in text.split() if word not in special_tokens])
        text = ''.join([i for i in text if not i.isdigit()])
        text = text.replace(r'\s+', ' ')
        text = ' '.join([i for i in text.split() if len(i)>1])
        
        text = text.replace(r'\s+', ' ')
        return text

def get_special_tokens(path="./special_words.txt"):
    """ 
    This method returns list of special words to be removed from the text.
    
    Input: path: str --> Path to the word file

    Output: : special_tokens: list[str] --> all words/special tokens to be removed
    """
    special_tokens=[]
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        special_tokens = f.readlines()
    special_tokens = [line.rstrip('\n') for line in special_tokens]
    
    return special_tokens

def get_clean_job_str(job_title, job_post):
    """ 
    This method returns cleaned Job Posting from Job Title and Job String. 
    Appends title and body tokens and concatenates the two.
    
    Input: Job Title Raw : string
            Job Body Raw : string
    Output: job_str: string:  cleaned and concatenated job details
    """
    title_token = "[TTL] "
    body_token = " [DESC] "

    job_title = clean_text(job_title, special_tokens=get_special_tokens())
    job_post = clean_text(job_post, special_tokens=get_special_tokens())

    job_str = title_token + job_title + body_token + job_post

    return job_str

def get_all_onets(onet_data_path="./data/raw/All_Occupations.csv"):
    """ 
    This method returns list of all ONETs available on the official site
    
    Input: path: str -->  (Optional) Path to the onet csv file

    Output: : all_onets_original: list[str] --> all ONETs available
    """
    if not os.path.exists(onet_data_path):
        print("No Onets found")
        return []
    print("All Onets found")
    all_occupations_df = pd.read_csv(onet_data_path)
    all_onets_original = all_occupations_df.Occupation.to_list()

    return all_onets_original


def get_onet_dicts(all_onets_original=get_all_onets()):
    """ 
    This method returns 2 dictionaries used to map standard ONET Names to string IDs. 
    
    Input: all_onets_original: list[str] --> (Optional) list of all ONETs

    Output: : id_to_onet_dict: dict[str, str] --> standard mapping of string id to ONETs --> "id" : "ONET_NAME"
              onet_to_id_dict: dict[str, str] --> standard mapping of ONETs to string id --> "ONET_NAME" : "id"
    """
    id_to_onet_dict = {str(id):onet for id, onet in enumerate(all_onets_original)}
    onet_to_id_dict = {onet:id for id,onet in id_to_onet_dict.items()}
    return id_to_onet_dict, onet_to_id_dict


def get_embd(input_docs, model=None, model_ckpt=None, save_embd=False, save_path=None):
    """ 
    This method computes embedding of the given string or list of strings using the given SBERT model. 
    
    Input: input_docs: str or list[str] --> list of input string. Can be a single string which will be converted to a list.
           model: SentenceTransformers() --> (Optional) pre-trained SBERT model, if None, checks for path 
           model_ckpt: str --> (Optional) pre-trained SBERT model checkpoint path, if None, loads basic SBERT from HF library 
           save_embd: Boolean --> (Optional) Flag to save computed embeddings.
           save_path: str --> (Optional) Path to save computed embeddings. If empty, new path will be created

    Output: simple_embd: numpy.ndarray (len(input), embd_size) --> array of Computed sentence embeddings. 
    """

    if not input_docs:
        return None
    if not isinstance(input_docs, list):
        print("convert string to list")
        input_docs = [input_docs]       

    if model:
        print("loading model as it is")
        sbert_model = model
    elif model_ckpt:
        print("loading model from ckpt")
        sbert_model = SentenceTransformer(model_ckpt)
    else:
        print("loading HF base model")
        sbert_model = SentenceTransformer("msmarco-distilbert-base-v4")
    
    simple_embd = sbert_model.encode(input_docs, show_progress_bar=True)
    
    if save_embd and save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            #os.chmod(save_path+"embd.pkl", 0o777)
        with open(save_path+"embd.pkl", "wb") as fOut:
            print("lenght of docs: ", len(input_docs))
            print("lenght of embd: ", len(simple_embd))
            pickle.dump({'input': input_docs, 'embeddings': simple_embd}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    return simple_embd

def get_onet_embeddings(model=None, model_ckpt=None, onet_embd_path=None, save_embd=False, save_path="./data/processed/embeddings/onet/"):
    """ 
    This method computes embedding for all ONETs available. 
    
    Input: model: SentenceTransformers() --> (Optional) pre-trained SBERT model, if None, checks for path 
           model_ckpt: str --> (Optional) pre-trained SBERT model checkpoint path, if None, loads basic SBERT from HF library 
           onet_embd_path: str --> (Optional) Path to load computed embeddings from pickle. If empty, Embeddings will be computed from scratch. 
           save_embd: Boolean --> (Optional) Flag to save computed embeddings.
           save_path: str --> (Optional) Path to save computed embeddings. If empty, new path will be created

    Output: onet_embd_df: pandas.DataFrame --> Dataframe with 2 columns:["ONET_NAME", "ONET_EMBD"] 
                                               Computed sentence embeddings can be stored as key, val pair. 
    """
    onet_embd_df = pd.DataFrame(columns=["ONET_NAME", "ONET_EMBD"])
    
    if not onet_embd_path:
        # Create onet embeddings from all onet data
        # Get list of all ONETs
        all_onets_original = get_all_onets()

        # Compute Embeddings of the entire list 
        simple_embd = get_embd(all_onets_original, model=model, model_ckpt=model_ckpt, save_embd=save_embd, save_path=save_path)

        # Save Embds as dataframe
        onet_embd_df["ONET_NAME"] = pd.Series(all_onets_original)
        onet_embd_df["ONET_EMBD"] = pd.Series([arr for arr in simple_embd])
        
    elif os.path.exists(onet_embd_path):
        #Load sentences & embeddings from disc
        with open(onet_embd_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            onet_embd_df["ONET_NAME"] = stored_data['input']
            onet_embd_df["ONET_EMBD"] = pd.Series([arr for arr in stored_data['embeddings']])
    
    print("Total ONETs available: ",len(onet_embd_df))
    return onet_embd_df

def get_job_embed_df_from_df(job_df=None,model=None, model_ckpt=None, job_embd_path=None, save_embd=False, save_path="./data/processed/embeddings/job/"):
    """ 
    This method computes embedding for all ONETs available. 
    
    Input: job_df: pandas.DataFrame --> (Optional) Raw job data Dataframe with at least 2 columns:["TITLE_RAW", "BODY"], if empty, loads precomputed.
           model: SentenceTransformers() --> (Optional) pre-trained SBERT model, if None, checks for path 
           model_ckpt: str --> (Optional) pre-trained SBERT model checkpoint path, if None, loads basic SBERT from HF library 
           job_embd_path: str --> (Optional) Path to load computed embeddings from pickle. If empty, Embeddings will be computed from scratch. 
           save_embd: Boolean --> (Optional) Flag to save computed embeddings.
           save_path: str --> (Optional) Path to save computed embeddings. If empty, new path will be created

    Output: onet_embd_df: pandas.DataFrame --> Dataframe with 2 columns:["ONET_NAME", "ONET_EMBD"] 
                                               Computed sentence embeddings can be stored as key, val pair. 
    """
    if job_df is None:
        print("Path to Job DF Given")
        job_df = pd.DataFrame(columns=["TITLE_RAW","BODY", "CLEANED_JOB", "JOB_EMBD"])
        # load embeddings from stored DF embeddings
        if os.path.exists(job_embd_path):
            #Load sentences & embeddings from disc
            with open(job_embd_path, "rb") as fIn:
                stored_data = pickle.load(fIn)
                # Loads cleaned job str and its embeddings
                job_df["CLEANED_JOB"] = stored_data['input']
                job_df["JOB_EMBD"] = pd.Series([arr for arr in stored_data['embeddings']])
                job_df["TITLE_RAW"] = job_df["CLEANED_JOB"].apply(lambda x:x[6:x.find(" [DESC] ")])
                job_df["BODY"] = job_df["CLEANED_JOB"].apply(lambda x:x[x.find(" [DESC] ")+1:])
        else:
            print("Path to Job DF does not exists")
            return job_df
    elif len(job_df) > 0:
        # DF present, compute from Raw DF
        if not ("TITLE_RAW" in job_df.columns and "BODY" in job_df.columns):
            print("Incomplete DataFrame, please try again")
            return None
        if not "CLEANED_JOB" in job_df.columns:
            job_df["CLEANED_JOB"] = job_df.apply(lambda x:get_clean_job_str(x["TITLE_RAW"], x["BODY"]), axis=1)
        
        simple_embd = get_embd(job_df["CLEANED_JOB"].to_list(), model=model, model_ckpt=model_ckpt, save_embd=save_embd, save_path=str(save_path)+str(len(job_df))+"/")

        job_df["JOB_EMBD"] = pd.Series([arr for arr in simple_embd])
    else:
        print("Unexpected Input Job DF, please try again")
    
    print("Total Jobs available: ",len(job_df)) 
    return job_df

def get_job_embd_df_frm_title_body(job_title, job_body, model=None, model_ckpt=None):
    job_df = pd.DataFrame({ "TITLE_RAW" : [job_title],
                            "BODY"      : [job_body], })
                            #"CLEANED_JOB": get_clean_job_str(job_title, job_body)
    job_df = get_job_embed_df_from_df(job_df=job_df, model=model, model_ckpt=model_ckpt)

    return job_df


