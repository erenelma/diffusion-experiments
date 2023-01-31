from scipy.special import factorial
import os.path
from re import M
import os
import math
import time
import re
import string
import gc
from itertools import product


from itertools import combinations
import sys
import random
import configparser


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


import pytorch_lightning as pl

from pytorch_lightning.callbacks import  ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch_geometric.loader as geom_loader
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from pytorch_lightning.loggers import CSVLogger

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from numpy import linalg as LA  




from models import *
from training_function import *
from thresholding_functions import *
from matrix_functions import * 
from utility_functions import * 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:21"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

PATH_CONFIG_FILE = sys.argv[1]


booleans=[]
floats=["lambda_try",  "dp_rate" , "val_set_ratio","learnr","weight_decay","threshold_value", "thresholding_step"]
ints = ["tay_step","c_hidden", "num_layers","threshold","max_epoch"]


config = configparser.ConfigParser()
config.read(PATH_CONFIG_FILE)
defaults = config['default']

variables = dict(defaults)

def fixVariableTypes(varDict):
  for k,v in varDict.items():
      if k in booleans:
          if varDict[k] == '1':
                  varDict[k]=True
          elif varDict[k] == '0':
                  varDict[k]=False
          else: sys.exit(f"Argument {varDict[k]} couldn't be parsed as Boolean.")


      elif k in floats:
          try:
              varDict[k] = float(varDict[k])
          except:
              sys.exit(f"Argument {varDict[k]} couldn't be parsed as float.")
      elif k in ints:
          try:
              varDict[k] = int(varDict[k])
          except:
              sys.exit(f"Argument {varDict[k]} couldn't be parsed as int.")
              
if __name__ == "__main__":
  PATH_DICT = {}
  
  fixVariableTypes(variables)
  
  
  
  
  variables['reg_str'] = "<" + variables['tag'] + ">(.*?)</" + variables['tag'] + ">"
  
  
  
  
  PATH_DICT["NAME_MODEL"]   = concatenate_strings([variables["model_name"], variables["layer_name"]])
  
  PATH_DICT["NAME_MATRIX"] = concatenate_strings([variables["name"], variables["matrix"], "lt", variables["lambda_try"], "ts", variables["tay_step"]])
  PATH_DICT["NAME_FULL"] = concatenate_strings([PATH_DICT["NAME_MATRIX"], "threshold",variables["threshold"], PATH_DICT["NAME_MODEL"]])
  PATH_DICT["NAME_FULL_WITHOUT_THRESHOLD"] = concatenate_strings([PATH_DICT["NAME_MATRIX"], PATH_DICT["NAME_MODEL"]])
  PATH_DICT["NAME_THRESHOLDED_MATRIX"] = concatenate_strings([PATH_DICT["NAME_MATRIX"], "threshold", variables["threshold"]])
  PATH_DICT["NAME_THRESHOLDED_MATRIX_INDEX"] = concatenate_strings([PATH_DICT["NAME_THRESHOLDED_MATRIX"], "index"])
  PATH_DICT["NAME_THRESHOLDED_MATRIX_EDGE_WEIGHT"] = concatenate_strings([PATH_DICT["NAME_THRESHOLDED_MATRIX"], "edge-weight"])
  PATH_DICT["NAME_NODE_FEATURE_MATRIX"] = concatenate_strings([PATH_DICT["NAME_MATRIX"], "node-feature-matrix"])
  
  
  
  
  PATH_DICT["PATH_ROOT"]      = os.getcwd() + "/"
  PATH_DICT["PATH_LOG_FOLDER"] = PATH_DICT["PATH_ROOT"] + "logs/"
  
  
  
  
  PATH_DICT["PATH_METRIC_SCORES_TRAIN_FOLDER"]                = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["metric_score_folder"], "train-scores/"])
  PATH_DICT["PATH_METRIC_SCORES_VAL_FOLDER"]                  = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["metric_score_folder"], "val-scores/"])
  PATH_DICT["PATH_METRIC_SCORES_TEST_FOLDER"]                 = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["metric_score_folder"], "test-scores/"])
  PATH_DICT["PATH_TRAIN_RESULTS_FOLDER"]                      = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["results_folder"], "train-results/"])
  PATH_DICT["PATH_VAL_RESULTS_FOLDER"]                        = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["results_folder"], "val-results/"])
  PATH_DICT["PATH_TEST_RESULTS_FOLDER"]                       = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["results_folder"], "test-results/"])
  PATH_DICT["PATH_DATASET_FOLDER"]                            = concatenate_strings([PATH_DICT["PATH_ROOT"], "datasets/"])
  
  
  
  PATH_DICT["PATH_THRESHOLDED_MATRIX"]             = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"], PATH_DICT["NAME_THRESHOLDED_MATRIX"] + ".torch"])
  PATH_DICT["PATH_THRESHOLDED_MATRIX_INDEX"]       = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"], PATH_DICT["NAME_THRESHOLDED_MATRIX_INDEX"] + ".torch"])
  PATH_DICT["PATH_THRESHOLDED_MATRIX_EDGE_WEIGHT"] = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"], PATH_DICT["NAME_THRESHOLDED_MATRIX_EDGE_WEIGHT"] + ".torch"])
  
  
  
  
  PATH_DICT["PATH_MATRIX"]                             = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["matrix_folder"], PATH_DICT["NAME_MATRIX"] + ".torch"])
  PATH_DICT["PATH_TRAIN_RESULTS"]                      = concatenate_strings([PATH_DICT["PATH_TRAIN_RESULTS_FOLDER"], PATH_DICT["NAME_FULL"] , "train-results.csv"])
  PATH_DICT["PATH_VAL_RESULTS"]                        = concatenate_strings([PATH_DICT["PATH_VAL_RESULTS_FOLDER"],   PATH_DICT["NAME_FULL"] , "val-results.csv"])
  PATH_DICT["PATH_TEST_RESULTS"]                       = concatenate_strings([PATH_DICT["PATH_TEST_RESULTS_FOLDER"],  PATH_DICT["NAME_FULL"] , "test-results.csv"])
  PATH_DICT["PATH_TEMP_METRICS_DF_TRAIN"]              = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["temp_metrics_folder"], PATH_DICT["NAME_FULL"] ,"train-temp-metrics.csv"])
  PATH_DICT["PATH_TEMP_METRICS_DF_VAL"]                = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["temp_metrics_folder"], PATH_DICT["NAME_FULL"] ,"val-temp-metrics.csv"])
  PATH_DICT["PATH_TEMP_METRICS_DF_TEST"]               = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["temp_metrics_folder"], PATH_DICT["NAME_FULL"] ,"test-temp-metrics.csv"])
  PATH_DICT["PATH_EXPERIMENT_INFO_DF"]                 = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["experiment_info_folder"], PATH_DICT["NAME_FULL"] ,"experiment-info.csv"])
  PATH_DICT["PATH_METRIC_SCORES_TRAIN"]                = concatenate_strings([PATH_DICT["PATH_METRIC_SCORES_TRAIN_FOLDER"], PATH_DICT["NAME_FULL"] ,"train-metric-score.csv"])
  PATH_DICT["PATH_METRIC_SCORES_VAL"]                  = concatenate_strings([PATH_DICT["PATH_METRIC_SCORES_VAL_FOLDER"], PATH_DICT["NAME_FULL"] ,"val-metric-score.csv"])
  PATH_DICT["PATH_METRIC_SCORES_TEST"]                 = concatenate_strings([PATH_DICT["PATH_METRIC_SCORES_TEST_FOLDER"], PATH_DICT["NAME_FULL"] ,"test-metric-score.csv"])
  PATH_DICT["PATH_TENSORBOARD_SAVE"]                   = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["tensorboard_save_folder"], PATH_DICT["NAME_FULL"]])
  PATH_DICT["PATH_NODE_FEATURE_MATRIX"]                = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["matrix_folder"], PATH_DICT["NAME_NODE_FEATURE_MATRIX"] ,".torch"])
  PATH_DICT["PATH_MODEL_CHECKPOINT"]                   = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["model_checkpoint_folder"]])
  PATH_DICT["PATH_CSV_LOGGER"]                         = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["csv_logger_folder"]])
  PATH_DICT["PATH_THVALUESEARCH_CHECKPOINT"]           = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thvalue_search_checkpoint_folder"]])
  PATH_DICT["PATH_THRESHOLD_SEARCHING_RESULTS"]        = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["threshold_searching_results_folder"], PATH_DICT["NAME_FULL_WITHOUT_THRESHOLD"], ".csv"])
  PATH_DICT["PATH_TUNE_RESULTS"]                       = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["tune_result_folder"], PATH_DICT["NAME_FULL"], "tune-results.csv"])
  PATH_DICT["PATH_DATASET"]                            = concatenate_strings([PATH_DICT["PATH_DATASET_FOLDER"], variables["name"]])
  PATH_DICT["PATH_THRESHOLDING_SPACE"]= concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholding_space_folder"], PATH_DICT["NAME_FULL_WITHOUT_THRESHOLD"], variables["thresholding_step"], "thresholding-space.torch"])
  
  
  
  all_dirs = [
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["matrix_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["tune_result_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["model_checkpoint_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["tensorboard_save_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["results_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["temp_metrics_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["experiment_info_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["metric_score_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["csv_logger_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thvalue_search_checkpoint_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["threshold_searching_results_folder"]]),
  concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholding_space_folder"]]),
  PATH_DICT["PATH_METRIC_SCORES_TRAIN_FOLDER"],
  PATH_DICT["PATH_METRIC_SCORES_VAL_FOLDER"],
  PATH_DICT["PATH_METRIC_SCORES_TEST_FOLDER"],
  PATH_DICT["PATH_TRAIN_RESULTS_FOLDER"],
  PATH_DICT["PATH_VAL_RESULTS_FOLDER"],
  PATH_DICT["PATH_TEST_RESULTS_FOLDER"]
  ]
  
  make_all_dirs(all_dirs)
  
  os.chdir('/content')
  
  variables['name_of_data_file'] = 'data'
  
  
  with open(PATH_DICT["PATH_DATASET"], "r") as f:
    senseval = f.read()
  
  variables['reg_str'] = "<" + variables['tag'] + ">(.*?)</" + variables['tag'] + ">"
  strs = re.findall(variables["reg_str"], senseval, re.DOTALL) # DOTALL matches '.' for ALL characters ( including '\n' )
  stop_words = set(stopwords.words('english'))
  texts = []
  for ctx in strs:
    ctx_content = (re.sub('<[^>]*>', '', ctx)) # removes all pos tags
    ctx_no_stopwords = [w for w in ctx_content.split() if w.lower() not in stop_words]
    ctx_clean = str(' ').join([w for w in ctx_no_stopwords if (w not in string.punctuation and w.isalpha())])
    texts.append(ctx_clean)
  
  
  labels = []
  
  for line in senseval.split('\n'):
      for w in line.split():
          if variables["label_tag"] in w and w.endswith("\"/>"):
              labels.append(w[w.find('"') + 1 : w.rfind('"')])
  classNames =  sorted(set(labels))
  ys = dict((j, i) for i, j in enumerate(classNames))
  y = np.array([ys[label] for label in labels]) # interest_1 -> 0, ...
  num_classes = len(classNames)
  
  texts_tokenized = [word_tokenize(w) for w in texts]
  texts_joined = [' '.join(ww) for ww in texts_tokenized]
  
  count_vectorizer = CountVectorizer()
  X = count_vectorizer.fit_transform(texts_joined)
  
  X_arr=X.toarray()
  
  label=y
  
  
  if variables["runmode"] == "training":
    hyperparameters = {
      "c_in":None,
      "c_out":len(classNames),
      "num_layers":variables["num_layers"],
      "c_hidden":variables["c_hidden"],
      "learnr":variables["learnr"],
      "dp_rate":variables["dp_rate"],
      "weight_decay":variables["weight_decay"],
      "threshold":variables["threshold"],
      "layer_name":variables["layer_name"],
      "model_name":variables["model_name"],
      "runmode":variables["runmode"],
    }
    
    label_ratio_list = split_string(variables["test_ratio_list"], one_minus=True)
    random_seed_list = split_string(variables["random_seed_list"], dtype=int)
    test_ratio_list  = split_string(variables["test_ratio_list"])       
    
    train_results_dataframe=load_dataframe(PATH_DICT["PATH_TRAIN_RESULTS"], cols = label_ratio_list, rows = random_seed_list)
    val_results_dataframe=load_dataframe(PATH_DICT["PATH_VAL_RESULTS"], cols = label_ratio_list, rows = random_seed_list)
    test_results_dataframe=load_dataframe(PATH_DICT["PATH_TEST_RESULTS"], cols = label_ratio_list, rows = random_seed_list)
    
    
    
    for test_r in test_ratio_list:
      label_ratio = round((1-test_r),4)
      row_counter=0
      if test_results_dataframe[label_ratio].notna().all():
        continue
      for rs in random_seed_list:
        if test_results_dataframe.loc[rs].notna()[label_ratio]:
          row_counter+=1
          continue
        
        
        
        creating_adj_start_time = time.time()
        Adj,node_feature_matrix = create_adj(X_arr = X_arr, device = device, PATH_DICT = PATH_DICT, variables = variables)
        creating_adj_end_time = time.time()
        creating_adj_time = creating_adj_end_time - creating_adj_start_time
        
        
        if variables["threshold_value"] != -1:
          threshold_value = variables["threshold_value"]
        else:
          threshold_value = None
        thresholding_start_time = time.time()
        threshold_value, num_edges, num_features = create_pytorch_data_object(Adj,node_feature_matrix=node_feature_matrix, threshold=hyperparameters["threshold"], rand_s=rs, test_ratio=test_r, threshold_value = threshold_value, PATH_DICT = PATH_DICT, variables = variables, label = label, device = device)
        thresholding_end_time = time.time()
        thresholding_time = thresholding_end_time - thresholding_start_time
        
        hyperparameters["c_in"] = num_features
        
        data_dif = create_text_data_object(variables = variables)
        del Adj
        gc.collect()
        torch.cuda.empty_cache()
        
        node_data_loader = DataLoader(data_dif, batch_size=1)
        del data_dif
        del node_feature_matrix
        gc.collect()
        torch.cuda.empty_cache()
        training_function_start_time = time.time()
        experiment_result=training(config=hyperparameters,              
                                  n_data_loader=node_data_loader,
                                  runmode=hyperparameters["runmode"],
                                  tstr=test_r,
                                  rand_s=rs,
                                  PATH_TENSORBOARD_SAVE=PATH_DICT["PATH_TENSORBOARD_SAVE"],
                                  PATH_CSV_LOGGER= PATH_DICT["PATH_CSV_LOGGER"],
                                  PATH_MODEL_CHECKPOINT= PATH_DICT["PATH_MODEL_CHECKPOINT"],
                                  FULL_NAME = PATH_DICT["NAME_FULL"],
                                  variables=variables,
                                  PATH_DICT=PATH_DICT)
                                      
        training_function_end_time = time.time()
        training_function_time = training_function_end_time - training_function_start_time
        
        
        
        temp_metrics_train = load_dataframe(PATH_DICT["PATH_TEMP_METRICS_DF_TRAIN"], cols = [k for k in experiment_result["classification_df_train"].columns])
        temp_metrics_val   = load_dataframe(PATH_DICT["PATH_TEMP_METRICS_DF_VAL"],   cols = [k for k in experiment_result["classification_df_val"].columns])
        temp_metrics_test  = load_dataframe(PATH_DICT["PATH_TEMP_METRICS_DF_TEST"],  cols = [k for k in experiment_result["classification_df_test"].columns])
        
        if temp_metrics_train.empty:
          temp_metrics_train = experiment_result["classification_df_train"]
        else:
          temp_metrics_train = temp_metrics_train.add(experiment_result["classification_df_train"])
        
        if temp_metrics_val.empty:
          temp_metrics_val = experiment_result["classification_df_val"]
        else:
          temp_metrics_val = temp_metrics_val.add(experiment_result["classification_df_val"])
        
        if temp_metrics_test.empty:
          temp_metrics_test = experiment_result["classification_df_test"]
        else:
          temp_metrics_test = temp_metrics_test.add(experiment_result["classification_df_test"])
          
          
        temp_metrics_train.to_csv(PATH_DICT["PATH_TEMP_METRICS_DF_TRAIN"], mode="w")
        temp_metrics_val.to_csv(PATH_DICT["PATH_TEMP_METRICS_DF_VAL"], mode="w")
        temp_metrics_test.to_csv(PATH_DICT["PATH_TEMP_METRICS_DF_TEST"], mode="w")
        
        
        train_results_dataframe.loc[rs][label_ratio] = experiment_result["train_acc"] 
        val_results_dataframe.loc[rs][label_ratio]   = experiment_result["val_acc"] 
        test_results_dataframe.loc[rs][label_ratio]  = experiment_result["test_acc"]
    
        train_results_dataframe.to_csv(PATH_DICT["PATH_TRAIN_RESULTS"])
        val_results_dataframe.to_csv(PATH_DICT["PATH_VAL_RESULTS"]) 
        test_results_dataframe.to_csv(PATH_DICT["PATH_TEST_RESULTS"])
    
        experiment_info_dict = {
        "experiment_time"          : [time.asctime(time.localtime(time.time()))],
        "dataset"                  : [variables["name"]],
        "c_in"                     : [len(texts_joined)],
        "c_out"                    : [len(classNames)],
        "c_hidden"                 : [hyperparameters["c_hidden"]],
        "num_layers"               : [hyperparameters["num_layers"]],
        "dp_rate"                  : [hyperparameters["dp_rate"]],
        "learnr"                   : [hyperparameters["learnr"]],
        "weight_decay"             : [hyperparameters["weight_decay"]],
        "threshold"                : [hyperparameters["threshold"]],
        "threshold_value"          : [threshold_value],
        "layer_name"               : [hyperparameters["layer_name"]],
        "model_name"               : [hyperparameters["model_name"]],
        "runmode"                  : [hyperparameters["runmode"]],
        "random_seed"              : [rs],
        "label_ratio"              : [label_ratio],
        "train_acc"                : [experiment_result["train_acc"]],
        "val_acc"                  : [experiment_result["val_acc"]],
        "test_acc"                 : [experiment_result["test_acc"]],
        "tay_step"                 : [variables["tay_step"]],
        "lambda_try"               : [variables["lambda_try"]],
        "max_epoch"                : [variables["max_epoch"]],
        "create_adj_time"          : [creating_adj_time],
        "thresholding_time"        : [thresholding_time],
        "train_function_time"      : [training_function_time],
        "fitting_time"             : [experiment_result["train_fitting_time"]],
        "edges"                    : [num_edges], 
        "training_nodes"           : [node_data_loader.dataset[0].train_mask.sum().item()],
        "test_nodes"               : [node_data_loader.dataset[0].test_mask.sum().item()],
        "validation_nodes"         : [node_data_loader.dataset[0].val_mask.sum().item()],
        }
        if variables["model_name"] == "GNN":
          experiment_info_dict["nodes"]                    = node_data_loader.dataset[0].num_nodes
          experiment_info_dict["avg_node_degree"]          = num_edges / node_data_loader.dataset[0].num_nodes
          experiment_info_dict["training_node_label_rate"] = (node_data_loader.dataset[0].train_mask.sum() / node_data_loader.dataset[0].num_nodes).item()
          experiment_info_dict["is_directed"]              = node_data_loader.dataset[0].is_directed()
          experiment_info_dict["self_loops"]               = node_data_loader.dataset[0].has_self_loops()
          experiment_info_dict["isolated_nodes"]           = node_data_loader.dataset[0].has_isolated_nodes() 
        cols_of_experiment_info_dict = [col for col in experiment_info_dict.keys()]
        if os.path.exists(PATH_DICT["PATH_EXPERIMENT_INFO_DF"]):
          experiment_info_df = pd.read_csv(PATH_DICT["PATH_EXPERIMENT_INFO_DF"], header=0, names=cols_of_experiment_info_dict)
          experiment_info_df = experiment_info_df.append(pd.DataFrame(experiment_info_dict, columns=cols_of_experiment_info_dict) , ignore_index = True)
        else:
          experiment_info_df = pd.DataFrame(experiment_info_dict)
        
        experiment_info_df.to_csv(PATH_DICT["PATH_EXPERIMENT_INFO_DF"], mode="w")
        row_counter+=1
      
      metrics_dataframe_train = temp_metrics_train.div(row_counter)
      metrics_dataframe_val   = temp_metrics_val.div(row_counter)
      metrics_dataframe_test  = temp_metrics_test.div(row_counter)
      
      with open(PATH_DICT["PATH_METRIC_SCORES_TRAIN"], mode="a") as f:
        f.write(f"label_ratio,{label_ratio}\n")
      with open(PATH_DICT["PATH_METRIC_SCORES_VAL"], mode="a") as f:
        f.write(f"label_ratio,{label_ratio}\n")
      with open(PATH_DICT["PATH_METRIC_SCORES_TEST"], mode="a") as f:
        f.write(f"label_ratio,{label_ratio}\n")
        
      metrics_dataframe_train.to_csv(PATH_DICT["PATH_METRIC_SCORES_TRAIN"], mode="a")
      metrics_dataframe_val.to_csv(PATH_DICT["PATH_METRIC_SCORES_VAL"], mode="a")
      metrics_dataframe_test.to_csv(PATH_DICT["PATH_METRIC_SCORES_TEST"], mode="a")
      
      os.remove(PATH_DICT["PATH_TEMP_METRICS_DF_TRAIN"])
      os.remove(PATH_DICT["PATH_TEMP_METRICS_DF_VAL"])
      os.remove(PATH_DICT["PATH_TEMP_METRICS_DF_TEST"])
      
      
    if "mean" not in train_results_dataframe.index:  
      train_results_dataframe.loc["mean"] = train_results_dataframe.mean(axis=0)
      val_results_dataframe.loc["mean"]   = val_results_dataframe.mean(axis=0)
      test_results_dataframe.loc["mean"]  = test_results_dataframe.mean(axis=0)
      
      train_results_dataframe.to_csv(PATH_DICT["PATH_TRAIN_RESULTS"],mode="w")
      val_results_dataframe.to_csv(PATH_DICT["PATH_VAL_RESULTS"],mode="w")
      test_results_dataframe.to_csv(PATH_DICT["PATH_TEST_RESULTS"],mode="w")
  elif variables["runmode"] == "tuning":
    hyperparameters = {
      "c_in":len(texts_joined),
      "c_out":len(classNames),
      "threshold":variables["threshold"],
      "layer_name":variables["layer_name"],
      "model_name":variables["model_name"],
      "runmode":variables["runmode"],
      "matrix_name":PATH_DICT["NAME_MATRIX"]
      }
  
    
    search_space={
        "num_layers"  : split_string(variables["num_layers_list"], dtype=int),
        "c_hidden"    : split_string(variables["c_hidden_list"], dtype=int),
        "learnr"      : split_string(variables["learnr_list"]),
        "dp_rate"     : split_string(variables["dp_rate_list"]),
        "weight_decay": split_string(variables["weight_decay_list"])                   
    }
    
    cols_of_tune_results = ["train_acc","val_acc","test_acc" ,"num_layers","c_hidden" ,"learnr","dp_rate","weight_decay","threshold","layer_name","model_name","c_in","c_out","lambda_try","tay_step","runmode","dataset","val-recall-macro","val-recall-micro","val-recall-weighted","val-precision-macro","val-precision-micro","val-precision-weighted","val-f1-macro","val-f1-micro","val-f1-weighted","test-recall-macro","test-recall-micro","test-recall-weighted" ,"test-precision-macro","test-precision-micro","test-precision-weighted","test-f1-macro","test-f1-micro" ,"test-f1-weighted", "train_fitting_time", "label_ratio", "random_seed", "memory_error", "max_gpu_alloc", "device","num_edges" ]
    
    
    
    
    
    
    tune_results_dataframe=load_dataframe(PATH_DICT["PATH_TUNE_RESULTS"], cols = cols_of_tune_results)
    
    test_ratio  = 0.95
    label_ratio = round((1-test_ratio),4) 
    rs = 42
    variables["max_epoch"] = 300
    
    for num_layers, c_hidden,  learnr, dp_rate, weight_decay in product(search_space["num_layers"], search_space["c_hidden"], search_space["learnr"], search_space["dp_rate"],  search_space["weight_decay"]):
      if ((tune_results_dataframe["num_layers"] == num_layers) & (tune_results_dataframe['c_hidden'] == c_hidden) & (tune_results_dataframe['learnr'] == learnr) & (tune_results_dataframe['dp_rate'] == dp_rate)& (tune_results_dataframe['weight_decay'] == weight_decay)).any():
        continue
      
      
      hyperparameters["num_layers"]   = num_layers
      hyperparameters["c_hidden"]     = c_hidden
      hyperparameters["learnr"]       = learnr
      hyperparameters["dp_rate"]      = dp_rate
      hyperparameters["weight_decay"] = weight_decay
      Adj,node_feature_matrix=create_adj(X_arr = X_arr, device = device, PATH_DICT = PATH_DICT, variables = variables)
      threshold_value, num_edges, num_features = create_pytorch_data_object(Adj=Adj, node_feature_matrix=node_feature_matrix, PATH_DICT = PATH_DICT, variables = variables, label = label, device = device, threshold=hyperparameters["threshold"],  rand_s=rs, test_ratio=test_ratio)
      hyperparameters["c_in"] = num_features
      
      
      data_dif = create_text_data_object(variables = variables)
      del Adj
      gc.collect()
      torch.cuda.empty_cache()
      
      node_data_loader = DataLoader(data_dif, batch_size=1)
      del data_dif
      gc.collect()
      torch.cuda.empty_cache()
      
      experiment_result=training(config=hyperparameters,              
                                 n_data_loader=node_data_loader,
                                 runmode=hyperparameters["runmode"],
                                 tstr=test_ratio,
                                 rand_s=rs,
                                 FULL_NAME = PATH_DICT["NAME_FULL"],
                                 variables=variables,
                                 PATH_DICT = PATH_DICT
                                  )
                                  
      tune_result_dict = {
      "train_acc"               : experiment_result["train_acc"],
      "val_acc"                 : experiment_result["val_acc"],
      "test_acc"                : experiment_result["test_acc"],
      "c_hidden"                : c_hidden,
      "num_layers"              : num_layers,
      "dp_rate"                 : dp_rate,
      "learnr"                  : learnr,
      "weight_decay"            : weight_decay,
      "threshold"               : variables["threshold"],
      "layer_name"              : variables["layer_name"],
      "model_name"              : variables["model_name"],
      "c_in"                    : node_data_loader.dataset[0].num_nodes,
      "c_out"                   : len(classNames),
      "lambda_try"              : variables["lambda_try"],
      "tay_step"                : variables["tay_step"],
      "runmode"                 : variables["runmode"],
      "dataset"                 : variables["name"],
      "val-recall-macro"        : experiment_result["val-recall-macro"],
      "val-recall-micro"        : experiment_result["val-recall-micro"],
      "val-recall-weighted"     : experiment_result["val-recall-weighted"],
      "val-precision-macro"     : experiment_result["val-precision-macro"],
      "val-precision-micro"     : experiment_result["val-precision-micro"],
      "val-precision-weighted"  : experiment_result["val-precision-weighted"],
      "val-f1-macro"            : experiment_result["val-f1-macro"],
      "val-f1-micro"            : experiment_result["val-f1-micro"],
      "val-f1-weighted"         : experiment_result["val-f1-weighted"],
      "test-recall-macro"       : experiment_result["test-recall-macro"],
      "test-recall-micro"       : experiment_result["test-recall-micro"],
      "test-recall-weighted"    : experiment_result["test-recall-weighted"],
      "test-precision-macro"    : experiment_result["test-precision-macro"],
      "test-precision-micro"    : experiment_result["test-precision-micro"],
      "test-precision-weighted" : experiment_result["test-precision-weighted"],
      "test-f1-macro"           : experiment_result["test-f1-macro"],
      "test-f1-micro"           : experiment_result["test-f1-micro"],
      "test-f1-weighted"        : experiment_result["test-f1-weighted"],
      "train_fitting_time"      : experiment_result["train_fitting_time"],
      "label_ratio"             : label_ratio,
      "random_seed"             : rs,
      "num_edges"               : num_edges
      }
      tune_results_dataframe = tune_results_dataframe.append(tune_result_dict, ignore_index=True)
      tune_results_dataframe.to_csv(PATH_DICT["PATH_TUNE_RESULTS"], mode="w")
  elif variables["runmode"] == "thresholdvalue-searching":
    hyperparameters = {
        "c_in":None,
        "c_out":len(classNames),
        "c_hidden":variables["c_hidden"],
        "num_layers":variables["num_layers"],
        "dp_rate":variables["dp_rate"],
        "learnr":variables["learnr"],
        "weight_decay":variables["weight_decay"],
        "threshold":variables["threshold"],
        "layer_name":variables["layer_name"],
        "model_name":variables["model_name"],
        "runmode":variables["runmode"]
    }
    
    temp_adj,_ = create_adj(X_arr = X_arr, device = device, PATH_DICT = PATH_DICT, variables = variables)
    my_ratio_space = create_ratio_space()
    if os.path.exists(PATH_DICT["PATH_THRESHOLDING_SPACE"]):
      my_thresholding_space = torch.load(PATH_DICT["PATH_THRESHOLDING_SPACE"])
    else:
      my_thresholding_space = create_thresholding_space(temp_adj, step=variables["thresholding_step"], ratio_space = my_ratio_space )
      torch.save(my_thresholding_space, PATH_DICT["PATH_THRESHOLDING_SPACE"])
    
    del temp_adj
    gc.collect()
    torch.cuda.empty_cache()
    
    test_ratio_list = split_string(variables["test_ratio_list"])
    label_ratio_list = split_string(variables["test_ratio_list"], one_minus=True)
    random_seed_list = split_string(variables["random_seed_list"], dtype=int)
    
    label_ratio_list_for_df = [f"test-acc-{element}" for element in label_ratio_list] + [f"train-acc-{element}" for element in label_ratio_list] + [f"val-acc-{element}" for element in label_ratio_list] + [f"val-f1-macro-{element}" for element in label_ratio_list]+ [f"val-f1-micro-{element}" for element in label_ratio_list] + [f"val-f1-weighted-{element}" for element in label_ratio_list] +  ["threshold_value", "num_edges","max_gpu_alloc", "fitting_time","device", "is_directed", "self_loops", "isolated_nodes" ]
    
    threshold_list_for_df=[]
    for k in my_thresholding_space:
      threshold_list_for_df.append(k[0])
      
    threshold_searching_results = load_dataframe(PATH_DICT["PATH_THRESHOLD_SEARCHING_RESULTS"], cols = label_ratio_list_for_df, rows = threshold_list_for_df)
    for temp_threshold_list in my_thresholding_space:
      
      temp_threshold = temp_threshold_list[0]
      temp_threshold_value= temp_threshold_list[1]
      temp_num_edges = temp_threshold_list[2]
      
      
      
      PATH_DICT["NAME_FULL"] = concatenate_strings([PATH_DICT["NAME_MATRIX"], "threshold",temp_threshold, PATH_DICT["NAME_MODEL"]])
      
      for test_r in test_ratio_list:
        label_ratio = round((1-test_r),4)
        if threshold_searching_results.loc[temp_threshold].notna()[f"test-acc-{label_ratio}"]:
          continue
        temp_train_accuracy=0
        temp_val_accuracy=0
        temp_test_accuracy=0
        random_seed_counter=0
        temp_f1_macro = 0
        temp_f1_micro = 0
        temp_f1_weighted =0
        temp_train_fitting_time = 0
        for rs in random_seed_list:
          Adj,node_feature_matrix=create_adj(X_arr = X_arr, device = device, PATH_DICT = PATH_DICT, variables = variables)
          threshold_value, num_edges, num_features= create_pytorch_data_object(Adj = Adj, node_feature_matrix=node_feature_matrix, PATH_DICT = PATH_DICT, variables = variables, label = label, device = device, threshold =temp_threshold ,  rand_s=rs, test_ratio=test_r, threshold_value = temp_threshold_value)
          if num_edges != temp_num_edges:
            sys.exit(0)
          hyperparameters["c_in"] = num_features
          data_dif = create_text_data_object(variables = variables)
          del Adj
          del node_feature_matrix
          gc.collect()
          torch.cuda.empty_cache()
          node_data_loader = DataLoader(data_dif, batch_size=1)
          flag_is_directed = node_data_loader.dataset[0].is_directed()
          flag_self_loops  = node_data_loader.dataset[0].has_self_loops()
          flag_isolated_nodes = node_data_loader.dataset[0].has_isolated_nodes()
          del data_dif
          gc.collect()
          torch.cuda.empty_cache()
          experiment_result=training(config=hyperparameters,              
                                    n_data_loader=node_data_loader,
                                    runmode=hyperparameters["runmode"],
                                    tstr=test_r,
                                    rand_s=rs,
                                    PATH_TENSORBOARD_SAVE = None,
                                    PATH_CSV_LOGGER = None,
                                    PATH_MODEL_CHECKPOINT = PATH_DICT["PATH_THVALUESEARCH_CHECKPOINT"],
                                    FULL_NAME = PATH_DICT["NAME_FULL"],
                                    variables = variables,
                                    PATH_DICT = PATH_DICT)
          temp_train_accuracy     += experiment_result["train_acc"]
          temp_val_accuracy       += experiment_result["val_acc"]
          temp_test_accuracy      += experiment_result["test_acc"]
          temp_f1_macro           += experiment_result["val-f1-macro"]
          temp_f1_micro           += experiment_result["val-f1-micro"]
          temp_f1_weighted        += experiment_result["val-f1-weighted"]
          temp_train_fitting_time += experiment_result["train_fitting_time"]
          random_seed_counter+=1
          del node_data_loader
          gc.collect()
          torch.cuda.empty_cache()
        mean_train_acc          = temp_train_accuracy/random_seed_counter
        mean_val_acc            = temp_val_accuracy/random_seed_counter
        mean_test_acc           = temp_test_accuracy/random_seed_counter
        mean_f1_macro           = temp_f1_macro/random_seed_counter
        mean_f1_micro           = temp_f1_micro/random_seed_counter
        mean_f1_weighted        = temp_f1_weighted/random_seed_counter
        mean_train_fitting_time = temp_train_fitting_time/random_seed_counter
        threshold_searching_results.loc[temp_threshold][f"train-acc-{label_ratio}"]       = mean_train_acc
        threshold_searching_results.loc[temp_threshold][f"val-acc-{label_ratio}"]         = mean_val_acc
        threshold_searching_results.loc[temp_threshold][f"test-acc-{label_ratio}"]        = mean_test_acc
        threshold_searching_results.loc[temp_threshold]["num_edges"]                      = num_edges
        threshold_searching_results.loc[temp_threshold]["threshold_value"]                = threshold_value
        threshold_searching_results.loc[temp_threshold][f"val-f1-macro-{label_ratio}"]    = mean_f1_macro
        threshold_searching_results.loc[temp_threshold][f"val-f1-micro-{label_ratio}"]    = mean_f1_micro
        threshold_searching_results.loc[temp_threshold][f"val-f1-weighted-{label_ratio}"] = mean_f1_weighted
        threshold_searching_results.loc[temp_threshold]["fitting_time"]                   = mean_train_fitting_time
        threshold_searching_results.loc[temp_threshold]["is_directed"]                    = flag_is_directed
        threshold_searching_results.loc[temp_threshold]["self_loops"]                     = flag_self_loops
        threshold_searching_results.loc[temp_threshold]["isolated_nodes"]                 = flag_isolated_nodes
        threshold_searching_results.to_csv(PATH_DICT["PATH_THRESHOLD_SEARCHING_RESULTS"], mode="w")
