import random
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import os
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch_geometric.loader as geom_loader
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def make_all_dirs(list_of_dirs):
  for dir in list_of_dirs:
    if not os.path.exists(dir):
      os.makedirs(dir)
      
def calc_all_metrics(truth_labels, predicted_labels):
  metric_results_dict = {
      "macro_results"    : score(y_true = truth_labels, y_pred = predicted_labels, average="macro", zero_division=0),
      "micro_results"    : score(y_true = truth_labels, y_pred = predicted_labels, average="micro", zero_division=0),
      "weighted_results" : score(y_true = truth_labels, y_pred = predicted_labels, average="weighted", zero_division=0)
  }
  return metric_results_dict
  
def concatenate_strings(str_list):
  result = str(str_list[0])
  for element in str_list[1:]:
    if not isinstance(element, str):
      element = str(element)
    if element[0] != "." and result[-1]!= "/":
      result += "-"
    result += element
  return result
  

def set_seeds(rand_s):
  pl.seed_everything(rand_s , workers=True)
  torch.manual_seed(rand_s)
  np.random.seed(rand_s)
  random.seed(rand_s)
  torch.cuda.manual_seed_all(rand_s)
  
  
def split_lab_ind(lab_arr, ind_arr, unique_label_list):
  dict_lab_index = {}
  for u in unique_label_list:
    dict_lab_index[u] = []
  for i in range(len(lab_arr)):
    dict_lab_index[lab_arr[i]].append(ind_arr[i])
  return dict_lab_index

def transfer_labels(source_lab, source_ind, target_lab, target_ind, changing_indexes):
  for index in changing_indexes:
    source_lab.append(target_lab[index])
    source_ind.append(target_ind[index])
    del target_ind[index]
    del target_lab[index]
  return source_lab, source_ind, target_lab, target_ind, len(changing_indexes) != 0

def reorganise_labeling(source_lab, source_ind, target_lab, target_ind, unique_label_list):
  source_lab_index = split_lab_ind(source_lab, source_ind, unique_label_list)
  target_lab_index = split_lab_ind(target_lab, target_ind, unique_label_list)
  changing_indexes = []
  for key in source_lab_index.keys():
    if len(source_lab_index[key]) == 0:
      if len(target_lab_index[key]) > 1:
        random_index = np.random.choice(target_lab_index[key],1)
        changing_indexes.append(target_ind.index(random_index.item()))
      assert len(target_lab_index[key]) > 1,  f"Number of label:{key} is 1 or less in the target_lab_index."
  return transfer_labels(source_lab, source_ind, target_lab, target_ind, changing_indexes)
  
  
def create_pytorch_data_object(Adj,node_feature_matrix, PATH_DICT, variables,  label, device, threshold,rand_s=42,test_ratio=0.2, threshold_value=None):
  
  if variables["runmode"] == "thresholdvalue-searching":
    PATH_DICT["NAME_THRESHOLDED_MATRIX"]             = concatenate_strings([PATH_DICT["NAME_MATRIX"], "threshold", threshold])
    PATH_DICT["NAME_THRESHOLDED_MATRIX_INDEX"]       = concatenate_strings([PATH_DICT["NAME_THRESHOLDED_MATRIX"], "index"])
    PATH_DICT["NAME_THRESHOLDED_MATRIX_EDGE_WEIGHT"] = concatenate_strings([PATH_DICT["NAME_THRESHOLDED_MATRIX"], "edge-weight"])
    PATH_DICT["PATH_THRESHOLDED_MATRIX"]             = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"], PATH_DICT["NAME_THRESHOLDED_MATRIX"] + ".torch"])
    PATH_DICT["PATH_THRESHOLDED_MATRIX_INDEX"]       = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"], PATH_DICT["NAME_THRESHOLDED_MATRIX_INDEX"] + ".torch"])
    PATH_DICT["PATH_THRESHOLDED_MATRIX_EDGE_WEIGHT"] = concatenate_strings([PATH_DICT["PATH_LOG_FOLDER"],variables["thresholded_matrix_folder"], PATH_DICT["NAME_THRESHOLDED_MATRIX_EDGE_WEIGHT"] + ".torch"])
  
  y=torch.tensor(label, dtype=torch.long, device=device)
  indices = np.arange(len(label))
  
  val_ratio = variables["val_set_ratio"]
  
  set_seeds(rand_s)
  train_lab, test_label, train_ind,test_index = train_test_split(label,indices, test_size=test_ratio, stratify=label, random_state=rand_s)
  test_lab, val_lab, test_ind, val_ind=train_test_split(test_label,test_index, test_size=val_ratio, stratify=test_label, random_state=rand_s)
  
  
  
  
  
  train_lab = train_lab.tolist()
  test_lab  = test_lab.tolist()
  train_ind = train_ind.tolist()
  test_ind  = test_ind.tolist()
  
  train_lab, train_ind, test_lab, test_ind, reorganise_flag = reorganise_labeling(train_lab, train_ind, test_lab, test_ind, list(sorted(set(label))))
  
  
  train_lab = np.array(train_lab)
  test_lab  = np.array(test_lab)
  train_ind = np.array(train_ind)
  test_ind  = np.array(test_ind)
  
  
  
  
  
  if threshold_value == None:
    threshold_value = np.sort(np.unique(Adj))[threshold]
  if os.path.exists(PATH_DICT["PATH_THRESHOLDED_MATRIX_INDEX"]) and os.path.exists(PATH_DICT["PATH_THRESHOLDED_MATRIX_EDGE_WEIGHT"]):
    Index = torch.load(PATH_DICT["PATH_THRESHOLDED_MATRIX_INDEX"])
    weight = torch.load(PATH_DICT["PATH_THRESHOLDED_MATRIX_EDGE_WEIGHT"])
    num_edges = 0
    for i in range(Index.shape[1]):
      if Index[0][i] <= Index[1][i]:
        num_edges +=1
  else:
    Index=[]
    weight=[]
    num_edges=0
    for i in range(Adj.shape[0]):
      for j in range(Adj.shape[1]):
        if Adj[i,j] >= threshold_value and Adj[i,j]>0: 
          Index.append([i,j])
          weight.append(Adj[i,j])
          if i<=j:
            num_edges+=1
    Index=np.ascontiguousarray(np.array(Index).T)
    weight=np.array(weight,dtype=np.float32)
    torch.save(Index ,  PATH_DICT["PATH_THRESHOLDED_MATRIX_INDEX"])
    torch.save(weight , PATH_DICT["PATH_THRESHOLDED_MATRIX_EDGE_WEIGHT"])
  
  edge_index = torch.tensor(Index, dtype=torch.long, device=device)
  
  if variables["layer_name"] == "GAT":
    weight = np.ascontiguousarray(weight.reshape(edge_index.shape[1], -1), dtype=np.float32)
  
  edge_weight=torch.tensor(weight, dtype=torch.float32, device=device)
  x = torch.tensor(node_feature_matrix , dtype=torch.float32, device=device)
  
    
    
  num_features = x.size(1)
  train_mask = torch.zeros(x.size(0), dtype=torch.bool)
  train_mask[torch.tensor(train_ind)] = True
    
  val_mask = torch.zeros(x.size(0), dtype=torch.bool)
  val_mask[torch.tensor(val_ind)] = True
    
  test_mask = torch.zeros(x.size(0), dtype=torch.bool)
  test_mask[torch.tensor(test_ind)] = True 
  
  
  data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, train_mask=train_mask, val_mask=val_mask , test_mask=test_mask).to(device)
  torch.save(data,  '/content/' + variables['name_of_data_file'] + '.pt')
  return threshold_value , num_edges , num_features
  
  
class TextData(Dataset):
  def __init__(self, data_file_name, transform=None, pre_transform=None):
        super(TextData, self).__init__(transform, pre_transform)
        self.data_file_name=data_file_name
  @property
  def raw_file_names(self): 
    pass

  @property
  def processed_file_names(self):
    return [self.data_file_name + '.pt']

  def __len__(self):
    return 1
  

  def len(self):
    return 1 

  def _download(self):
        pass

  def _process(self):
    pass
     
  def get(self, idx) -> Data :
    data=torch.load('/content/' + self.data_file_name + '.pt' )    
    return data
  
  
  def __getitem__(self, idx):
        data=torch.load('/content/' + self.data_file_name + '.pt' )
        return data
        

def create_text_data_object(variables): 
  data_dif=TextData(variables['name_of_data_file'])
  return data_dif
  
def load_dataframe(path, cols = None, rows = None):
  if os.path.exists(path):
    return pd.read_csv(path, header=0, names=cols)
  else:
    return pd.DataFrame(columns=cols, index=rows)
  
def split_string(input, target_list=None, delimiter=",", dtype=float, one_minus=False , extras=None, rounding=4):
  if target_list is not None:
    output = target_list[:]
  else:
    output=[]
  if dtype == float:
    if one_minus == False:
      output.extend([(float(splitted)) for splitted in input.split(delimiter)])
    elif one_minus == True:
      output.extend([round((1-float(splitted)),rounding) for splitted in input.split(delimiter)])
  elif dtype == int:
    if one_minus == False:    
      output.extend([(int(splitted)) for splitted in input.split(delimiter)])
    elif one_minus == True:
      output.extend([(1-int(splitted)) for splitted in input.split(delimiter)])
  if extras is not None:
    output.extend(extras)      
  return output