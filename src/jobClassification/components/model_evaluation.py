import os
from jobClassification.logging import logger
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation, util, CrossEncoder
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import pandas as pd

from jobClassification.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
      
    def calculate_cosine_scores(self, evaluator):
        model = CrossEncoder(self.config.tokenizer_path)
        cross_scores = model.predict([[ doc, query] for query, doc in zip(evaluator.sentences1, evaluator.sentences2)])
        # normalize to 0 to 1
        rescaled_array = (cross_scores-np.min(cross_scores))/(np.max(cross_scores)-np.min(cross_scores))
        rescaled_array = np.round(rescaled_array,2)

        return rescaled_array

    def calculate_accuracy(self, predicted_scores, actual_scores):
        errors = []
        accuracy = 0
        for i, score in enumerate(actual_scores):
            error = abs(score - predicted_scores[i] * 100)
            errors.append(error)
        if(len(errors)) == 0:
            logger.info("No sentences to predict : ")
        else:
            accuracy = 100 - sum(errors)/len(errors)
            logger.info(f"Val accuracy with cross-encoders : {accuracy}" )
        return accuracy


    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sbert_model = SentenceTransformer(self.config.model_path).to(device)
        
        #loading data 
        
        val_dataloader = torch.load(os.path.join(self.config.data_path,"val.pth"))
        test_dataloader = torch.load(os.path.join(self.config.data_path,"test.pth"))

        val_evaluator = torch.load(os.path.join(self.config.data_path,"val_eval.pth"))
        test_evaluator = torch.load(os.path.join(self.config.data_path,"test_eval.pth"))

        predicted_score = self.calculate_cosine_scores(test_evaluator)
        accuray = self.calculate_accuracy(predicted_score, test_evaluator.scores)

        test_df = pd.DataFrame({
            "CLEAN_JOB"    :   test_evaluator.sentences1,
            "ONET_NAME"  :   test_evaluator.sentences2,
            "Actual_scores" :   test_evaluator.scores,
            "Actual_scores" :   predicted_score.tolist()
        })
        test_df.to_csv(self.config.metric_file_name, index=False)