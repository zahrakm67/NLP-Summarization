import pandas as pd
from thext import SentenceRankerPlus
from thext import DatasetPlus
from thext import Highlighter
import nltk
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

class FinetuneWithContext():
  
  def __init__(self,
                raw_dataset=None,
                dataset_name=None,
                model_name_or_path=None,
                base_model_name=None,
                context_builder_model=None,
                checkpoint_dir=None):
      self.raw_dataset = raw_dataset
      self.dataset_name = dataset_name
      self.base_model_name = base_model_name
      self.model_name_or_path = model_name_or_path
      self.context_builder_model=context_builder_model
      self.checkpoint_dir=checkpoint_dir


  def finetune(self):

    logging.info('----1-prepare_train_data')
    data_set_train=self.prepare_train_data()
    
    logging.info('----2-prepare_validation_data')
    data_set_validation=self.prepare_validation_data()

    logging.info('----3-SentenceRankerPlus_loading')
    sen_rank_model = SentenceRankerPlus(base_model_name=self.base_model_name,
                                        model_name_or_path=self.model_name_or_path,
                                        epochs=2)
    
    sen_rank_model.set_train_set(data_set_train)
    sen_rank_model.set_eval_set(data_set_validation)

    logging.info('----4-prepare_for_training')
    sen_rank_model.load_model(self.base_model_name)
    sen_rank_model.prepare_for_training()

    logging.info('----5-start_finetuning')
    sen_rank_model.fit(checkpoint_dir=self.checkpoint_dir)



  def prepare_train_data(self):
    train_df=self.raw_dataset["train"][:800]
    text=[]
    highlight=[]
    abstract=[]

    # Get text and highlights from related different
    if self.dataset_name=='cnn_dailymail':
      text=train_df['article']
      highlight=[ [item] for item in train_df['highlights'] ]
    elif self.dataset_name=='xsum':
      text=train_df['document']
      highlight=[ [item] for item in train_df['summary'] ]

    if self.context_builder_model!=None:
      abstract = self.build_context(model_name=self.context_builder_model, list_text=text)
    else:
      abstract = text

    data_set=DatasetPlus(list_text=text,
                       list_abstract=abstract,
                       list_highlights=highlight,
                       n_jobs=1)
    return data_set


  def prepare_validation_data(self):
    validation_df=self.raw_dataset["validation"][:200]
    text=[]
    highlight=[]
    abstract=[]

    # Get text and highlights from related different
    if self.dataset_name=='cnn_dailymail':
      text=validation_df['article']
      highlight=[ [item] for item in validation_df['highlights'] ]

    elif self.dataset_name=='xsum':
      text=validation_df['document']
      highlight=[ [item] for item in validation_df['summary'] ]

    if self.context_builder_model!=None:
      abstract = self.build_context(model_name=self.context_builder_model, list_text=text)
    else:
      abstract = text

    data_set=DatasetPlus(list_text=text,
                           list_abstract=abstract,
                           list_highlights=highlight,
                           n_jobs=1)

    return data_set


  def build_context(self, model_name, list_text):
      # model_name: saeedehj/t5-small-finetune-cnn, saeedehj/led-base-finetune-cnn, saeedehj/t5-small-finetune-xsum, saeedehj/led-base-finetune-xsum

      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
      abstract = []

      for i in range(len(list_text)):
          
          inputs = tokenizer(list_text[i], return_tensors="pt", truncation=True)
          outputs = model.generate(**inputs)
          summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)
          abstract.append(summary[0])

      return abstract



