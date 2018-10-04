# -*- coding: utf-8 -*-
"""
Created on Thu May 03 13:24:07 2018

@author: alexandre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()

path = r"C:\Users\alexandre\Documents\kaggle\rstudio\data"

##
def import_data():
    
    train = pd.read_csv(path + "/train.csv", 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    
    test = pd.read_csv(path + "/test.csv", 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    
    return train, test


if __name__ == "__main__":    
    train, test = import_data()