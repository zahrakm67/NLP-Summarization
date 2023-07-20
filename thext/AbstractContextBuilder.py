import nltk
import os
import gc
import datasets
from datasets import load_dataset, Dataset
from evaluate import load
import numpy as np
import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)



class AbstractContextBuilder():

    def __init__(self,
                 dataset_name=None,
                 model_checkpoint=None,
                 max_input_length=1024,
                 max_target_length=256,
                 epochs=20,
                 batch_size=4,
                 output_dir=None):

        logging.info("Abstract Context Builder - Initialization")

        self.epochs = epochs
        self.batch_size = batch_size

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.output_dir = output_dir

       
        # load dataset
        self.dataset_name = dataset_name
        self.dataset = self.load_data()
        
        # Model loading
        self.model_checkpoint = model_checkpoint
        self.prefix = ""  

        if self.model_checkpoint in ["t5-small", "t5-base", "t5-larg", "google/long-t5-local-base"]:
            self.prefix = "summarize: "

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)  
    
    def preprocess_function(self, dataset):
      
      if self.dataset_name == 'cnn_dailymail':
      
        inputs = [self.prefix + doc for doc in dataset["article"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)
        labels = self.tokenizer(text_target=dataset["highlights"], max_length=self.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

      elif self.dataset_name == 'xsum':
        inputs = [self.prefix + doc for doc in dataset["document"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)
        labels = self.tokenizer(text_target=dataset["summary"], max_length=self.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
   
    def load_data(self):
      if self.dataset_name == 'cnn_dailymail':
        return load_dataset(self.dataset_name, "3.0.0")
      else:
        return load_dataset(self.dataset_name)

    
    def train(self):
        
        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True)

        args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            fp16=True,
            output_dir=self.output_dir)

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=datasets.DatasetDict({"train": Dataset.from_dict(tokenized_datasets["train"][2000:10000])})["train"],
            eval_dataset=datasets.DatasetDict({"validation": Dataset.from_dict(tokenized_datasets["validation"][400:2000])})["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.save_model(self.model)

        # trainer.push_to_hub()


    def compute_metrics(self, eval_pred):
      predictions, labels = eval_pred
      metric = load("rouge")
      decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
      # Replace -100 in the labels as we can't decode them.
      labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
      decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

      # Rouge expects a newline after each sentence
      decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
      decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

      # and thus will return a list, computing a metric for each sentence.
      result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
      result = {key: value * 100 for key, value in result.items()}

      # Add mean generated length
      prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
      result["gen_len"] = np.mean(prediction_lens)

      return {k: round(v, 4) for k, v in result.items()}
    
    def save_model(self, model):
        model.save_pretrained(self.output_dir + '/model')


    def generate_summary(self, article):
        inputs = self.tokenizer(article, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

