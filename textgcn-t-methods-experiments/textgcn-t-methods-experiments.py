import os
import gc
import torch
import re
import nltk
from nltk.corpus import stopwords
import torch
import numpy as np
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import numpy as np
from scipy.special import factorial
from numpy import linalg as LA
import pandas as pd
from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from itertools import combinations
import sys
import math
import random
import configparser
import time
from re import M
from sklearn.model_selection import train_test_split

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



nltk.download('stopwords')
nltk.download('punkt')

CONFIG_FILE_NAME = sys.argv[1]



def set_seeds(rand_s):
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

def create_inv_degree_matrix(adj):
    degree_matrix=torch.zeros((adj.shape[0] , adj.shape[1]),dtype=torch.float32).to(device)
    i=0
    while i < adj.shape[1]:
      degree_matrix[i,i]=(1/torch.sum(adj[i]))
      i+=1
    return degree_matrix

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

def find_cooccurences(vocab, texts_tokenized):

    n_i  = OrderedDict((name, 0) for name in vocab)
    word2index = OrderedDict( (name,index) for index,name in enumerate(vocab) )
    occurrences = np.zeros( (len(vocab),len(vocab)) ,dtype=np.int32)
    
    no_windows = 0
    for l in tqdm(texts_tokenized, total=len(texts_tokenized)):
        if len(l)-variables["window"]>=0: 
            for i in range(len(l)-variables["window"]+1): 
                d = set(l[i:(i+variables["window"])])
                if d.issubset(n_i.keys()):
                    no_windows += 1
                    for w in d:
                        n_i[w] += 1
                    for w1,w2 in combinations(d,2):
                        i1 = word2index[w1]
                        i2 = word2index[w2]
                        occurrences[i1][i2] += 1
                        occurrences[i2][i1] += 1
        else:
            d = set(l)
            if d.issubset(n_i.keys()):
                no_windows += 1
                for w in d:
                    n_i[w] += 1
                for w1,w2 in combinations(d,2):
                    i1 = word2index[w1]
                    i2 = word2index[w2]
                    occurrences[i1][i2] += 1
                    occurrences[i2][i1] += 1
    return occurrences, n_i, no_windows



def calc_pij_pi2(vocab, occurrences, n_i, no_windows):
  p_ij=np.zeros(occurrences.shape)
  p_i=np.zeros(len(n_i.keys()))
  for i in range(len(p_ij)):
    p_i[i] = n_i[vocab[i]]/no_windows
    for j in range(len(p_ij[0])):
      p_ij[i,j] = occurrences[i,j]/no_windows
  return p_ij , p_i
      

def pmi_conversion2(p_ij,p_i):
  pmi_result=np.zeros(p_ij.shape)
  for i in range(len(p_ij)):
    for j in range(len(p_ij[0])):
      if p_ij[i,j] == 0:
        continue
      else:
        pmi_result[i,j]=np.log(p_ij[i,j]/(p_i[i] * p_i[j]))
      
  return pmi_result



def pmi(vocab, texts_tokenized):
    occurrences, n_i, no_windows = find_cooccurences(vocab, texts_tokenized)
    p_ij, p_i = calc_pij_pi2(vocab, occurrences, n_i, no_windows)
    p_ij = pmi_conversion2(p_ij, p_i)
    return p_ij


def Taylor_Expansion(lambda_try, G_matrix, tay_step):
        S_matrix = np.identity(np.size(G_matrix, axis=1))
        S_matrix = 2 * S_matrix
        for i in range(1, tay_step):
            S_matrix = S_matrix +  ((np.power(lambda_try, i) / factorial(i)) * (LA.matrix_power(G_matrix, i)))
        S_matrix = np.multiply(0.5, S_matrix)
        return S_matrix


def Normalization(S_matrix):
        N = (np.size(S_matrix, axis=1))
        Normalized_Kernel = np.zeros((N, N))
        M = np.size(Normalized_Kernel, axis=1)
        for i in range(0, M):
            for j in range(0, M):

                Normalized_Kernel[i, j] = abs(S_matrix[i, j]) / (np.sqrt(abs(S_matrix[i, i]) *abs(S_matrix[j, j]) ) )

        return Normalized_Kernel

def VNormalization(S_matrix):
    D = np.eye(S_matrix.shape[0])*np.sum(S_matrix, axis=1)
    DInverseRoot = np.sqrt(np.linalg.pinv(D))
    result = np.dot(DInverseRoot, np.dot(S_matrix, DInverseRoot))
    return result



def make_all_dirs(list_of_dirs):
  for dir in list_of_dirs:
    if not os.path.exists(dir):
      os.makedirs(dir)





booleans=[]
floats=["lambda_try","val_set_ratio", "learnr"]
ints = ["tay_step", "window", "hidden_size_1", "max_epoch"]



config = configparser.ConfigParser()
config.read(CONFIG_FILE_NAME)
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
fixVariableTypes(variables)









def concatenate_strings(str_list):
  result = str(str_list[0])
  for element in str_list[1:]:
    if not isinstance(element, str):
      element = str(element)
    if element[0] != "." and result[-1]!= "/":
      result += "-"
    result += element
  return result
      
def define_paths():
  
  global DATASET_NAME
  global MATRIX_NAME 
  global FULL_NAME  
  global PATH_MATRIX 
  global PATH_TRAIN_RESULTS
  global PATH_VAL_RESULTS
  global PATH_TEST_RESULTS
  global PATH_TRAIN_METRIC_SCORE
  global PATH_VAL_METRIC_SCORE
  global PATH_TEST_METRIC_SCORE
  global PATH_TIME_LOGGER
  global PATH_ROOT
  global PATH_DATASET
  
  
  PATH_ROOT               = os.getcwd() + "/"
  DATASET_NAME            = variables["name"]
  MATRIX_NAME             = concatenate_strings([DATASET_NAME, variables["matrix"], "lt",variables["lambda_try"],"ts",variables["tay_step"]])
  FULL_NAME               = concatenate_strings([MATRIX_NAME, "t"])
  
  PATH_MATRIX             = concatenate_strings([PATH_ROOT,variables["matrixpath"], MATRIX_NAME , ".torch"])
  PATH_TRAIN_RESULTS      = concatenate_strings([PATH_ROOT,variables["resultspath"], FULL_NAME, "train-results.csv"])
  PATH_VAL_RESULTS        = concatenate_strings([PATH_ROOT,variables["resultspath"], FULL_NAME, "val-results.csv"])
  PATH_TEST_RESULTS       = concatenate_strings([PATH_ROOT,variables["resultspath"], FULL_NAME, "test-results.csv"])
  PATH_TRAIN_METRIC_SCORE = concatenate_strings([PATH_ROOT,variables["metricscorespath"], FULL_NAME, "train-metrics.csv"])
  PATH_VAL_METRIC_SCORE   = concatenate_strings([PATH_ROOT,variables["metricscorespath"], FULL_NAME, "val-metrics.csv"])
  PATH_TEST_METRIC_SCORE  = concatenate_strings([PATH_ROOT,variables["metricscorespath"], FULL_NAME, "test-metrics.csv"])
  PATH_TIME_LOGGER        = concatenate_strings([PATH_ROOT,variables["timelogger"], FULL_NAME, "time-logger.csv"])
  PATH_DATASET            = concatenate_strings([PATH_ROOT,"datasets/", DATASET_NAME])


define_paths()

make_all_dirs([
PATH_ROOT+variables["matrixpath"],
PATH_ROOT+variables["resultspath"],
PATH_ROOT+variables["metricscorespath"],
PATH_ROOT+variables["timelogger"]])



def calc_a_hat(A):
    D_inv_sqrt= torch.sqrt(create_inv_degree_matrix(A))
    return torch.mm(torch.mm(D_inv_sqrt,A),D_inv_sqrt).to(device,dtype=torch.float32)

class gcn(nn.Module):
    
    def __init__(self, X_size, num_classes):
        super(gcn, self).__init__()
        self.fc1=nn.Linear(X_size,variables["hidden_size_1"])
        self.relu1 = nn.ReLU()
        self.fc2=nn.Linear(variables["hidden_size_1"] , num_classes)
        self.softmax2=nn.Softmax(dim=1)
        
    def forward(self,A_hat, X=None, identity_feature_matrix=False): 
        if identity_feature_matrix == False:
          out=torch.mm(A_hat,X)
          out=self.fc1(A_hat)
        else:
          out=self.fc1(A_hat)
        out=self.relu1(out)
        out=torch.mm(A_hat,out)
        out=self.fc2(out)
        out=self.softmax2(out)
        return out

def evaluate(output, labels_e):
    if len(labels_e) == 0:
        return 0
    else:
        _, labels = output.max(1);  
        labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        return sum([(e) for e in labels_e] == labels)/len(labels)


def calc_optim_adj(length, df_tfidf, p_ij):
    p_ij = np.bmat([[np.identity(length), df_tfidf], [df_tfidf.T, p_ij]])
    p_ij = np.array(p_ij)
    return p_ij



with open(PATH_DATASET, "r") as f:
  senseval = f.read()


variables['reg_str'] = "<" + variables['tag'] + ">(.*?)</" + variables['tag'] + ">"
strs = re.findall(variables["reg_str"], senseval, re.DOTALL) 
stop_words = set(stopwords.words('english'))
texts = []
for ctx in strs:
  ctx_content = (re.sub('<[^>]*>', '', ctx)) 
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
y = np.array([ys[label] for label in labels]) 
num_classes = len(classNames)

texts_tokenized = [word_tokenize(w) for w in texts]
texts_joined = [' '.join(ww) for ww in texts_tokenized]



def create_A():
  vectorizer = TfidfVectorizer(input="content")
  df_tfidf = vectorizer.fit_transform(texts_joined)
  df_tfidf = df_tfidf.toarray()
  vocab = vectorizer.get_feature_names_out()
  vocab = np.array(vocab)
  count_vectorizer = CountVectorizer(vocabulary=vocab) 
  X = count_vectorizer.fit_transform(texts_joined)
  X_arr=X.toarray()
  
  
  
  if os.path.exists(PATH_MATRIX):
    gc.collect()
    torch.cuda.empty_cache()
    wordbyword=torch.load(PATH_MATRIX)
  else:
    if variables["matrix"]=="TEXT-GCN":
      gc.collect()
      torch.cuda.empty_cache()
      wordbyword = pmi(vocab, texts_tokenized)
      wordbyword[wordbyword < 0] = 0
      np.fill_diagonal(wordbyword,1.0)
      torch.save(wordbyword, PATH_MATRIX)
    else:
      X_arr_tensor=torch.tensor(X_arr , dtype=torch.float, device=device)
      del X_arr
      del X
      del vocab
      gc.collect()
      torch.cuda.empty_cache()
      G_mat = torch.mm(torch.transpose(X_arr_tensor,0,1),X_arr_tensor).to(device)
      del X_arr_tensor
      gc.collect()
      torch.cuda.empty_cache()
      if isinstance(G_mat , torch.Tensor):
        G_mat = G_mat.cpu().detach().numpy()
      wordbyword = Taylor_Expansion(variables["lambda_try"], G_mat, variables["tay_step"])
      
      if variables["matrix"]=="T-DIF-NORM":
        wordbyword = Normalization(wordbyword)
  
      elif variables["matrix"]=="T-DIF-REG":
        wordbyword = VNormalization(wordbyword)
      elif variables["matrix"]=="T-DIF-NORM-REG":
        wordbyword = Normalization(wordbyword)
        wordbyword = VNormalization(wordbyword)
        
        
      torch.save(wordbyword, PATH_MATRIX)  
      del G_mat
      gc.collect()
      torch.cuda.empty_cache()
  wordbyword = calc_optim_adj(len(texts), df_tfidf, wordbyword)
  del df_tfidf
  gc.collect()
  torch.cuda.empty_cache()
  A=torch.tensor(wordbyword, dtype=torch.float32, device=device)
  del wordbyword
  gc.collect()
  torch.cuda.empty_cache()
  return A
  
def text_gcn(A, learn_r , test_ratio, rand_s=42):
      label_ratio = round(1-test_ratio, 4)
      calc_ahat_start_time = time.time()
      A_hat = calc_a_hat(A).to(device)
      calc_ahat_end_time = time.time()
      calc_ahat_time = calc_ahat_end_time - calc_ahat_start_time
      
      del A
      gc.collect()
      torch.cuda.empty_cache()
      f = torch.eye(A_hat.shape[0], device=device, dtype=torch.float32) 
      indices = np.arange(len(y))
      val_ratio = variables["val_set_ratio"]
      
      set_seeds(rand_s)
      train_lab, test_label, train_ind,test_index = train_test_split(y,indices, test_size=test_ratio, stratify=y, random_state=rand_s)
      test_lab, val_lab, test_ind, val_ind=train_test_split(test_label,test_index, test_size=val_ratio, stratify=test_label, random_state=rand_s)
      
      train_lab, test_lab, train_ind, test_ind = train_lab.tolist(), test_lab.tolist(), train_ind.tolist(), test_ind.tolist()
      
      train_lab, train_ind, test_lab, test_ind, reorganise_flag = reorganise_labeling(train_lab, train_ind, test_lab, test_ind, list(sorted(set(y))))

      train_lab_tensor=torch.tensor(train_lab , dtype=torch.long, device=device)
      
      set_seeds(rand_s)
      
      net = gcn(A_hat.shape[0], num_classes=num_classes).to(device)
  
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(net.parameters(), lr=learn_r)
      

      net.train()
      best_predictions_train=None
      best_predictions_val=None
      best_predictions_test=None
      max_val_acc  = 0
      max_test_acc = 0
      max_train_acc = 0
      max_val_acc_epoch=0
      training_start_time = time.time()
      for e in range(variables["max_epoch"]):
        optimizer.zero_grad()
        output = net(A_hat, identity_feature_matrix = True)
        loss = criterion(output[train_ind], train_lab_tensor)
        loss.backward()
        optimizer.step()
        pred_train = evaluate(output[train_ind] , train_lab)
        pred_val   = evaluate(output[val_ind] , val_lab)
        pred_test  = evaluate(output[test_ind], test_lab)
        if pred_val >= max_val_acc:
          max_train_acc =  pred_train
          max_val_acc = pred_val
          max_test_acc  =  pred_test
          max_val_acc_epoch = e
          best_predictions_train=output[train_ind]
          best_predictions_val=output[val_ind]
          best_predictions_test=output[test_ind]
      training_end_time = time.time()
      training_time = training_end_time - training_start_time
     
      
      classif_report_train = sklearn.metrics.classification_report(y_true = train_lab, y_pred = best_predictions_train.argmax(dim=1).detach().cpu().numpy() , zero_division=0 , output_dict=True)
      classif_report_val = sklearn.metrics.classification_report(y_true = val_lab, y_pred = best_predictions_val.argmax(dim=1).detach().cpu().numpy() , zero_division=0 , output_dict=True)
      classif_report_test = sklearn.metrics.classification_report(y_true = test_lab, y_pred = best_predictions_test.argmax(dim=1).detach().cpu().numpy() , zero_division=0 , output_dict=True)
      
      
      with open(PATH_TRAIN_METRIC_SCORE, "a") as fd:
        fd.write(f"rs,{rand_s}\n")
        fd.write(f"label_ratio,{label_ratio}\n")
      with open(PATH_VAL_METRIC_SCORE, "a") as fd:
        fd.write(f"rs,{rand_s}\n")
        fd.write(f"label_ratio,{label_ratio}\n")
      with open(PATH_TEST_METRIC_SCORE, "a") as fd:
        fd.write(f"rs,{rand_s}\n")
        fd.write(f"label_ratio,{label_ratio}\n")
      
      pd.DataFrame(classif_report_train).to_csv(PATH_TRAIN_METRIC_SCORE, mode="a")
      pd.DataFrame(classif_report_val).to_csv(PATH_VAL_METRIC_SCORE, mode="a")
      pd.DataFrame(classif_report_test).to_csv(PATH_TEST_METRIC_SCORE, mode="a")
      
      
      del net
      del train_lab_tensor
      del output
      del best_predictions_train
      del best_predictions_val
      del best_predictions_test
      del optimizer
      del A_hat
      del f
      del criterion
      gc.collect()
      torch.cuda.empty_cache()
      return max_train_acc, max_val_acc, max_test_acc, training_time, calc_ahat_time
      

random_seed_list = split_string(input = variables["random_seed_list"], dtype = int) 
label_ratio_list = split_string(input = variables["test_ratio_list"], one_minus=True)
test_ratio_list = split_string(input = variables["test_ratio_list"])


learnr = variables["learnr"]


train_results_df = load_dataframe(PATH_TRAIN_RESULTS, cols = label_ratio_list, rows = random_seed_list)
val_results_df = load_dataframe(PATH_VAL_RESULTS, cols = label_ratio_list, rows = random_seed_list)
test_results_df = load_dataframe(PATH_TEST_RESULTS, cols = label_ratio_list, rows = random_seed_list)
for rs in random_seed_list:
  for testr in test_ratio_list:
    label_ratio = round(1-testr,4)
    if test_results_df.loc[rs].notna()[label_ratio]:
      continue
    creating_adj_start_time = time.time()
    A = create_A()
    
    creating_adj_end_time = time.time()
    creating_adj_time = creating_adj_end_time - creating_adj_start_time
    train_acc, val_acc, test_acc, training_time, calc_ahat_time = text_gcn(A=A, rand_s=rs ,learn_r=learnr , test_ratio = testr)
    train_results_df.loc[rs][label_ratio] = train_acc
    val_results_df.loc[rs][label_ratio] = val_acc
    test_results_df.loc[rs][label_ratio] = test_acc
    train_results_df.to_csv(PATH_TRAIN_RESULTS,mode="w")
    val_results_df.to_csv(PATH_VAL_RESULTS,mode="w")
    test_results_df.to_csv(PATH_TEST_RESULTS,mode="w")
    del A
    gc.collect()
    torch.cuda.empty_cache()
    time_logger_df = load_dataframe(PATH_TIME_LOGGER, cols = ["creating_adj_time", "training_time",  "calc_ahat_time"])
    time_logger_df = time_logger_df.append(pd.DataFrame({"creating_adj_time":[creating_adj_time], "training_time":[training_time], "calc_ahat_time":[calc_ahat_time]}), ignore_index = True)
    time_logger_df.to_csv(PATH_TIME_LOGGER, mode="w")
train_results_df.loc["mean"]=train_results_df.mean()
val_results_df.loc["mean"]=val_results_df.mean()
test_results_df.loc["mean"]=test_results_df.mean()
