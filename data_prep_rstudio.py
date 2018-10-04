# -*- coding: utf-8 -*-
"""
Created on Thu May 03 13:24:07 2018

@author: alexandre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections, re
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
from pandas.io.json import json_normalize
from datetime import datetime
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()

path = r"C:\Users\alexandre\Documents\kaggle\rstudio\data"

def import_data():
    train = pd.read_csv(path + "/extracted_fields_train.gz", 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    
    test = pd.read_csv(path + "/extracted_fields_test.gz", 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    
    sample = pd.read_csv(path + "/sample_submission.csv") 
    return train, test, sample


def suppress_cols(train, test):
    sup = []
    for col in train.columns:
        if col in test.columns:
            if len(train[col].unique().tolist()) == 1:
                del train[col]
                del test[col]
                sup.append(col)
    print("cols suppressed are {0}".format(sup))
    
    return train, test 


def source_check(train, test):
    
   def number_points(x):
       return len(x.split("."))
       
   def correct_source(x): 
       x = x.split(":",1)[0]
       
       v = x.split(".", 1)
       if len(v[0]) <=2:
           if len(v)>1:
               x = v[1]
               
       if x.split(".")[-1] in ["jp", "pl", "ru", "vk", "tb", "uk"]:
           x = ".".join(x.split(".")[:-1])
       return x
   
   def dummify(x, words):
       out = [0]*len(words)
       for i in x.split("."):
           if i in words:
              out[words.index(i)] = 1
       return [out]
    
   ### get number of dots
   train["number_dots_source"] = train["trafficSource.source"].apply(lambda x :number_points(x))
   test["number_dots_source"] = test["trafficSource.source"].apply(lambda x :number_points(x))
   
   ### correct source
   train["trafficSource.source"] = train["trafficSource.source"].apply(lambda x : correct_source(x))
   test["trafficSource.source"] = test["trafficSource.source"].apply(lambda x : correct_source(x))

   collec = collections.Counter([y for x in  train["trafficSource.source"].tolist() + test["trafficSource.source"].tolist() for y in x.split(".")])
   counts = pd.DataFrame(np.transpose([collec.keys(), collec.values()]), columns = ["key", "value"])
   counts["value"] = counts["value"].astype(int)
   counts = counts.sort_values("value")
   words  = counts.loc[counts["value"]>150, "key"].tolist()
   
   ### dummify source
   train["dummies_source"] = train["trafficSource.source"].apply(lambda x : dummify(x, words))
   test["dummies_source"] = test["trafficSource.source"].apply(lambda x : dummify(x, words))
   
   tr = pd.DataFrame(np.array(list(zip(*train["dummies_source"])))[0], columns = words)
   train2 = pd.concat([train, tr], axis = 1)
   
   tt = pd.DataFrame(np.array(list(zip(*test["dummies_source"])))[0], columns = words)
   test2 = pd.concat([test, tt], axis= 1)
   
   del train2["trafficSource.source"]
   del test2["trafficSource.source"]
   
   return train2, test2


def create_date_vars(data):
    
    data["month"] = data["visitStartTime"].dt.month
    data["day_month"] = data["visitStartTime"].dt.day
    data["day_week"] = data["visitStartTime"].dt.dayofweek
    data["hour"] = data["visitStartTime"].dt.hour
    
    return data


def prep(train, test):
    
    to_sup = ['AW - Electronics',
             'Retail (DO NOT EDIT owners nophakun and tianyu)',
             'AW - Apparel',
             'All Products',
             'Data Share']
    
    train["trafficSource.campaign"] = np.where(train["trafficSource.campaign"].isin(to_sup), "Oth", train["trafficSource.campaign"])
    test["trafficSource.campaign"] = np.where(test["trafficSource.campaign"].isin(to_sup), "Oth", test["trafficSource.campaign"])
    
    train["trafficSource.medium"] = np.where(train["trafficSource.medium"] == "(not set)", "(none)", train["trafficSource.medium"])
    test["trafficSource.medium"] = np.where(test["trafficSource.medium"]  == "(not set)", "(none)", test["trafficSource.medium"])
    
    del train["trafficSource.isTrueDirect"]
    del test["trafficSource.isTrueDirect"]
    
    ### transaction revenues
    train["geoNetwork.continent"] = np.where(train["geoNetwork.continent"] == "(not set)", "Americas", train["geoNetwork.continent"])
    test["geoNetwork.continent"] = np.where(test["geoNetwork.continent"]  == "(not set)", "Americas", test["geoNetwork.continent"])
    
    ### get rid of 2 outliers
    train = train.loc[~train["fullVisitorId"].isin(["1957458976293878100", "0824839726118485274"])]
    
    ### visit time
    train["visitStartTime"] = train["visitStartTime"].apply(lambda x :datetime.utcfromtimestamp(x))
    test["visitStartTime"] = test["visitStartTime"].apply(lambda x : datetime.utcfromtimestamp(x))
    
    train = create_date_vars(train)
    test = create_date_vars(test)
    
    for col in ['channelGrouping', 
       u'device.browser',
       u'device.deviceCategory', u'device.operatingSystem',
       u'geoNetwork.city',  u'geoNetwork.country',
       u'geoNetwork.metro', u'geoNetwork.networkDomain', u'geoNetwork.region',
       u'geoNetwork.subContinent']:
        
        var_vs_target(train, 'totals.transactionRevenue', col, bins=30)
    
    train = train.drop([u'fullVisitorId',"date", u'sessionId', u'visitId', "visitStartTime"])
    
    return train, test
    

def var_vs_target(data, Y_label, variable, bins=30, normalize = False):
    
    if type(Y_label) == str:
        Y_label = [Y_label]
        
    data = data.copy()
    
    if variable not in data.columns:
        return "variable not in database"
   
    if len(data[variable].value_counts())>bins:
        if data[variable].dtype !="O":
            data[variable] = pd.qcut(data[variable] , bins, precision = 1, duplicates = "drop")
        else:
            modalities = data[variable].value_counts().index[:bins]
            data.loc[~data[variable].isin(modalities), variable] = "other"
        
    avg_target = data[Y_label].mean()
    if normalize:
        Y = data[[variable] + list(Y_label)].groupby(variable).mean() / data[list(Y_label)].mean()
        
    else:
        Y = data[[variable] + list(Y_label)].groupby(variable).mean()
        
    P = data[[variable] + list(Y_label)].groupby(variable).agg([np.size, np.std])
    
    ### add confidence_interval
    plt.figure(figsize= (12,8))
    
    ax1 = P[Y_label[0]]["size"].plot(kind="bar", alpha= 0.42, grid= True)
    ax2 = ax1.twinx()
    
    if normalize:
        ax2.set_ylim([np.min(np.min(Y))*0.95, np.max(np.max(Y))*1.05])
    
    s = ax2.plot(ax1.get_xticks(), Y[Y_label], linestyle='-', label= [Y_label])
    
    ax1.set_ylabel('%s Volume'%str(variable))
    ax2.set_ylabel('%s'%str(Y_label))
    ax1.set_xlabel('%s'%str(variable))
    
    plt.title("Evolution of %s vs %s"%(variable, Y_label))
    ax2.legend(tuple(Y_label), loc= 1, borderaxespad=0.)
    
    if not normalize:
        for i, value in enumerate(avg_target):
            plt.axhline(y=value, xmin=0, xmax=3, linewidth=0.5, linestyle="--", color = s[i].get_color())
            plt.errorbar(ax1.get_xticks(), Y[Y_label[i]], yerr=1.96*P[Y_label[i]]["std"]/np.sqrt(P[Y_label[0]]["size"]), alpha= 0.65, color= s[i].get_color())
    
    plt.setp(ax1.xaxis.get_ticklabels(), rotation=78)
    plt.show()
    
    return {"size/std" : P, "mean" : Y}

if __name__ == "__main__":    
    train, test, sample = import_data()
#    train, test = suppress_cols(train, test)
   