import os
from jobClassification.logging import logger

from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation, util
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
import tqdm

from jobClassification.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sbert_model = SentenceTransformer(self.config.model_ckpt).to(device)
        
        word_embedding_model = sbert_model._first_module()
        word_embedding_model.tokenizer.add_tokens(self.config.special_tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        loss = losses.CosineSimilarityLoss(model=sbert_model)

        #loading data 
        #dataset_processed = load_from_disk(self.config.data_path)


        train_dataloader = torch.load(os.path.join(self.config.data_path,"train.pth"))#DataLoader(train_examples, shuffle=True, batch_size=16)
        val_dataloader = torch.load(os.path.join(self.config.data_path,"val.pth"))
        test_dataloader = torch.load(os.path.join(self.config.data_path,"test.pth"))

        val_evaluator = torch.load(os.path.join(self.config.data_path,"val_eval.pth"))
        test_evaluator = torch.load(os.path.join(self.config.data_path,"test_eval.pth"))

        sbert_model.fit(train_objectives=[(test_dataloader, loss)], 
                        #evaluator=val_evaluator, 
                        epochs = self.config.num_train_epochs,
                        warmup_steps= self.config.warmup_steps, 
                        weight_decay= self.config.weight_decay, 
                        output_path= os.path.join(self.config.root_dir,"saved-search-model")
                    )