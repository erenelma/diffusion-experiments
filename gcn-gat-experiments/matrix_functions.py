from scipy.special import factorial
import os.path
import os
import time
import gc
import torch
import numpy as np
from numpy import linalg as LA



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
    
def sync_adj(matrix_a):
  for i in range(matrix_a.shape[0]):
    for j in range(i):
      matrix_a[i,j] = matrix_a[j,i]
  return matrix_a
  
def is_undirected(matrix_a):
  for i in range(matrix_a.shape[0]):
    for j in range(matrix_a.shape[1]):
      if matrix_a[i,j] != matrix_a[j,i]:
        return False
  return True
  
def count_num_edges_undirected(matrix_a, threshold_v=0.0):
  if threshold_v == 0:
    return np.count_nonzero(np.triu(matrix_a) > 0)
  elif threshold_v > 0:
    return np.count_nonzero(np.triu(matrix_a) >= threshold_v)
    
    
def create_adj(X_arr, device, PATH_DICT, variables):
    X_arr_tensor=torch.tensor(X_arr , dtype=torch.float64, device=device)
    
    if os.path.exists(PATH_DICT["PATH_MATRIX"]):
        Adj = torch.load(PATH_DICT["PATH_MATRIX"])
        if os.path.exists(PATH_DICT["PATH_NODE_FEATURE_MATRIX"]):
          node_feature_matrix = torch.load(PATH_DICT["PATH_NODE_FEATURE_MATRIX"])
        else:
          node_feature_matrix = None

        del X_arr_tensor
        gc.collect()
        torch.cuda.empty_cache()
        return Adj, node_feature_matrix
    else:
        if variables["matrix"] == "MATRIX-BOW":
            variables['name_of_data_file'] = 'METHOD-BOW-' + variables['name_of_data_file']
            
            Adj = torch.mm(X_arr_tensor , torch.transpose(X_arr_tensor , 0 , 1)).to(device)
            node_feature_matrix = X_arr_tensor
            if isinstance(Adj , torch.Tensor):
              Adj = Adj.cpu().detach().numpy()
            if isinstance(node_feature_matrix, torch.Tensor):
              node_feature_matrix = node_feature_matrix.cpu().detach().numpy()  
            
            
            
            torch.save(Adj,PATH_DICT["PATH_MATRIX"])
            torch.save(node_feature_matrix, PATH_DICT["PATH_NODE_FEATURE_MATRIX"])
          
        elif variables["matrix"] == "MATRIX-DIF-NORM":
            G_mat = torch.mm(torch.transpose(X_arr_tensor,0,1),X_arr_tensor)
            if isinstance(G_mat , torch.Tensor):
              G_mat = G_mat.cpu().detach().numpy()
            S_matrix = Taylor_Expansion(variables["lambda_try"], G_mat, variables["tay_step"])
            variables['name_of_data_file'] = 'METHOD-DIF-NORM-' + variables['name_of_data_file']
            
            if isinstance(S_matrix , torch.Tensor):
              S_matrix = S_matrix.cpu().detach().numpy()
            
            S_matrix = Normalization(S_matrix)
            
            S_matrix_tensor= torch.tensor(S_matrix , dtype=torch.float64).to(device)
            
            
            Adj = torch.mm( torch.mm(X_arr_tensor , S_matrix_tensor) ,
                           torch.mm(torch.transpose(S_matrix_tensor ,0,1), torch.transpose(X_arr_tensor,0,1)
                           )).to(device)
            node_feature_matrix = torch.mm(X_arr_tensor, S_matrix_tensor).to(device)
            
            if isinstance(Adj , torch.Tensor):
              Adj = Adj.cpu().detach().numpy()
            if isinstance(node_feature_matrix, torch.Tensor):
              node_feature_matrix = node_feature_matrix.cpu().detach().numpy()
            
            Adj = VNormalization(Adj)
            
            Adj = sync_adj(Adj)
            
            torch.save(Adj,PATH_DICT["PATH_MATRIX"])
            torch.save(node_feature_matrix, PATH_DICT["PATH_NODE_FEATURE_MATRIX"])
            
            del G_mat
            del S_matrix_tensor
        
        del X_arr_tensor
        gc.collect()
        torch.cuda.empty_cache()
        return Adj, node_feature_matrix
