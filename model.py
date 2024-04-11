import os 
os.environ["KERAS_BACKEND"] = "torch"

import seaborn as sns
import pandas as pd
import numpy as np

import torch
import keras
import keras_nlp

from sklearn.model_selection import train_test_split

class model_config:
    seed  = 42
    preset = "gemma_1.1_instruct_2b_en" # name of pretrained backbone
    train_seq_len = 1024 # max size of input sequence for training
    train_batch_size = 8 # size of the input batch in training
    infer_seq_len = 2000 # max size of input sequence for inference
    infer_batch_size = 2 # size of the input batch in inference
    epochs = 6 # number of epochs to train
    lr_mode = "exp" # lr scheduler mode from one of "cos", "step", "exp"
    
    labels = ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
              "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
              "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM",
              "I-STREET_ADDRESS","I-URL_PERSONAL","O"]
    id2label = dict(enumerate(labels)) # integer label to BIO format label mapping
    label2id = {v:k for k,v in id2label.items()} # BIO format label to integer label mapping
    num_labels = len(labels) # number of PII (NER) tags
    
    train = True # whether to train or use already trained ckpt

keras.utils.set_random_seed(model_config.seed)
keras.mixed_precision.set_global_policy("mixed_float16") 

BASE_PATH = 'data/'

