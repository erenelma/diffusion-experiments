import math
import sys
import numpy as np
from matrix_functions import is_undirected, count_num_edges_undirected


def thresholding_pool(matrix_a, step):
  max_edges = count_num_edges_undirected(matrix_a,0.0)
  possible_threshold_values = np.sort(np.unique(matrix_a))
  num_possible_threshold_values = possible_threshold_values.shape[0]
  increment = math.ceil(num_possible_threshold_values * step)
  pool=[]
  for threshold_index in range(0, num_possible_threshold_values, increment):
    temp_threshold_value = possible_threshold_values[threshold_index]
    num_edges = count_num_edges_undirected(matrix_a, temp_threshold_value)
    pool.append([threshold_index, temp_threshold_value, num_edges])
  return pool
  
def create_thresholding_space(matrix_a, step, ratio_space):
  max_num_edges  = count_num_edges_undirected(matrix_a)
  pool = thresholding_pool(matrix_a, step)
  ratio_space = sorted(ratio_space)
  thresholding_space = []
  for ratio in ratio_space:
    searched_num_edges = math.ceil(max_num_edges * ratio)
    for row in reversed(pool):
      if row[2] >= searched_num_edges:
        thresholding_space.append(row)
        break
  return thresholding_space
  
def create_ratio_space():
  search_ratio = 0.01
  search_ratio_space=[]
  while search_ratio <= 1.0:
    search_ratio = round(search_ratio,4)
    search_ratio_space.append(search_ratio) 
    if search_ratio < 0.1:
      search_ratio += 0.005
    elif search_ratio >= 0.1 and search_ratio < 0.2:
      search_ratio += 0.02
    elif search_ratio >= 0.2 and search_ratio < 0.6:
      search_ratio += 0.1
    elif search_ratio >= 0.6:
      search_ratio += 0.2
  return search_ratio_space