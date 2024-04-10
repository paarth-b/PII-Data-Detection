import os 
os.environ["KERAS_BACKEND"] = "torch"

import seaborn as sns
import pandas as pd
import torch
import keras
import keras_nlp

from sklearn.model_selection import train_test_split

class model_config:
    seed  = 42
    