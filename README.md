# Enhancing News Highlights Extraction through Abstractive Context 


This repository contains the code for our adaptation of the paper [Transformer-based Highlights Extraction from scientific papers - THExt](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006931), to the domain of news articles (cnn-dailymail, xsum). 


The process of extracting highlights involves selecting the most important sentences from a text that effectively summarize its meaning. This paper focuses on the problem of extracting highlights from news articles using transformer-based techniques.

Our task was to adapt THExt to another domain and enhance it by providing a new context. We propose a new approach that improves upon the baseline by generating context using the T5 and LongFormer architecture. The results obtained in the news domain, using two distinct datasets (cnn-dailymail, xsum), have demonstrated the effectiveness of the model in this field and its adaptability to different domains. 
Furthermore, we show that providing a broader context can further enhance the model's performance.


### 2. Abstractive Context Builder

The T5 model was utilized to generate a new context by performing abstractive summarization of articles from two datasets. This new context was then used in conjunction with the THExt model to extract highlights.

The pipeline for the Abstractive Context Builder is illustrated below:

<div align="center">
  <img src="https://github.com/saeedehj/THExt_Extended/blob/main/imgs/model.jpg" alt="Alt text" title="Abstractive Context Builder pipeline" width="600" height="200">
</div>

<p>


### Dependencies
* Python 3 (tested on python 3.6)
* PyTorch
  * with GPU and CUDA enabled installation (though the code is not runnable on CPU)
* TensorFlow
* pyrouge (for evaluation)

### Dataset Download 
The dataset exploited are the following: 
1. [xsum dataset](https://huggingface.co/datasets/xsum)
2. [cnn_dailymail dataset](https://huggingface.co/datasets/cnn_dailymail)

### Pre-trained model Download 
We finetuned T5-small and LongFormer models on xsum and cnn_dailymail datasets for extracting summaries, which were then used as context to improve the performance of the THExt model. The pre-trained models can be downloaded from the following links:

1. [xsum dataset]
  (https://huggingface.co/saeedehj/t5-small-finetune-xsum)
  (https://huggingface.co/saeedehj/led-base-finetune-xsum)
  
2. [cnn_dailymail dataset]
  (https://huggingface.co/saeedehj/t5-small-finetune-cnn)
  (https://huggingface.co/saeedehj/led-base-finetune-cnn)

 
