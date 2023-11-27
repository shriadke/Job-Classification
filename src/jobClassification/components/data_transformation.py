import os
from jobClassification.logging import logger
from jobClassification.entity import DataTransformationConfig
from sentence_transformers import InputExample, evaluation
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import torch
from torch.utils.data import DataLoader

import pandas as pd
from jobClassification.utils.common import get_clean_job_str
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def get_dataloader(self, data, split):
        examples = []
        data = data[split]
        n_examples = data.num_rows

        for i in range(n_examples):
            example = data[i]
            examples.append(InputExample(texts=[example['CLEAN_JOB'], example['ONET_NAME']], label=float(1)))
        logger.info(f"in {split}, We have a {type(examples)} of length {len(examples)} containing {type(examples[0])}'s.")
        dataloader = DataLoader(examples, shuffle=True, batch_size=16)
        return examples, dataloader

    def convert(self):

        # get csv data to df
        train_df = pd.read_csv(self.config.data_path+"raw_train/train_data.csv").head()
        print("TITLE_RAW"  in train_df.columns)
        test_df = pd.read_csv(self.config.data_path+"raw_test/test_data.csv").head()

        # add clean col to df
        train_df["CLEAN_JOB"] = train_df.apply(lambda x:get_clean_job_str(x["TITLE_RAW"], x["BODY"]), axis=1)
        test_df["CLEAN_JOB"] = test_df.apply(lambda x:get_clean_job_str(x["TITLE_RAW"], x["BODY"]), axis=1)

        # split train/val/test data
        train_ratio = 0.85
        val_ratio = 0.15
        train_df, val_df = train_test_split(train_df, test_size=1 - train_ratio, random_state=42, shuffle=True)

        final_data = DatasetDict({
            "train" : Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"]),
            "val" : Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"]),
            "test" : Dataset.from_pandas(test_df)
        })
        #final_data.save_to_disk( os.path.join(self.config.data_path,"final_data/"))

        dataset = final_data#load_from_disk( os.path.join(self.config.data_path,"final_data/"))

        
        train_examples, train_dataloader = self.get_dataloader(dataset, "train")
        torch.save(train_dataloader, os.path.join(self.config.root_dir,"train.pth"))

        
        val_examples, val_dataloader = self.get_dataloader(dataset, "val")
        val_evaluator = evaluation.EmbeddingSimilarityEvaluator([],[],[]).from_input_examples(examples=val_examples)
        torch.save(val_dataloader, os.path.join(self.config.root_dir,"val.pth"))
        torch.save(val_evaluator, os.path.join(self.config.root_dir,"val_eval.pth"))

        test_examples, test_dataloader = self.get_dataloader(dataset, "test")
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator([],[],[]).from_input_examples(examples=test_examples)
        torch.save(test_dataloader, os.path.join(self.config.root_dir,"test.pth"))
        torch.save(test_evaluator, os.path.join(self.config.root_dir,"test_eval.pth"))
