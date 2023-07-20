from thext import SentenceRankerPlus
from thext import Highlighter
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def prepare_evaluation(list_predicted_highlights, list_real_highlights): 
      
      scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
      scores = []

      for i in range(len(list_predicted_highlights)):
          scores.append(scorer.score(list_predicted_highlights[i], list_real_highlights[i]))

      return scores

def load_model(base_model_name, model_name_or_path):
    sr = SentenceRankerPlus()
    sr.load_model(base_model_name=base_model_name, model_name_or_path=model_name_or_path)
    h = Highlighter(sr)

    return h

def evaluate(base_model_name, model_name_or_path, raw_dataset, dataset_name, context_builder_model=None):
    sr = SentenceRankerPlus()
    sr.load_model(base_model_name=base_model_name, model_name_or_path=model_name_or_path)
    h = Highlighter(sr)

    train_df=raw_dataset["test"][:1000]  
    text=[]
    real_highlight=[]

    if dataset_name=='cnn_dailymail':
      text=train_df['article']
      real_highlight=[item for item in train_df['highlights']]
    elif dataset_name=='xsum':
      text=train_df['document']
      real_highlight=[item for item in train_df['summary']]

    predicted_highlights=[]
    predicted_highlights_concat=[]
    abstract=[]

    if context_builder_model!=None:
      abstract = build_context(model_name=context_builder_model, list_text=text)
    else:
      abstract = text
    
    
    for i in range(len(text)):
      highlights = h.get_highlights_simple(sent_tokenize(text[i]), abstract[i],
                        rel_w=1.0, 
                        pos_w=0.0, 
                        red_w=0.0, 
                        prefilter=False, 
                        NH = 3)
      predicted_highlights.append(highlights)

    for highlight in predicted_highlights:  
        predicted_highlights_concat.append(' '.join(map(str,highlight)))

    
    scores = prepare_evaluation(predicted_highlights_concat, real_highlight)
    average_scores(scores=scores)

def average_scores(scores):
    avg_precision = {}
    avg_recall = {}
    avg_fmeasure = {}

    # Iterate over the list of dictionaries
    for item in scores:
        # Iterate over the metrics
        for metric, score in item.items():
            # Calculate the sum for each metric
            avg_precision[metric] = avg_precision.get(metric, 0) + score.precision
            avg_recall[metric] = avg_recall.get(metric, 0) + score.recall
            avg_fmeasure[metric] = avg_fmeasure.get(metric, 0) + score.fmeasure

    # Calculate the average for each metric
    num_items = len(scores)
    avg_precision = {metric: value / num_items for metric, value in avg_precision.items()}
    avg_recall = {metric: value / num_items for metric, value in avg_recall.items()}
    avg_fmeasure = {metric: value / num_items for metric, value in avg_fmeasure.items()}

    # Print the average values
    logging.info("Average Precision:")
    logging.info(avg_precision)
    logging.info("Average Recall:")
    logging.info(avg_recall)
    logging.info("Average F-measure:")
    logging.info(avg_fmeasure)

def build_context(model_name, list_text):
    # model_name: saeedehj/t5-small-finetune-cnn, saeedehj/led-base-finetune-cnn, saeedehj/t5-small-finetune-xsum, saeedehj/led-base-finetune-xsum

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    abstract = []

    for i in range(len(list_text)):
        logging.info('Context generated from model')
        inputs = tokenizer(list_text[i], return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs)
        summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        abstract.append(summary[0])

    return abstract