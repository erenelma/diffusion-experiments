import time
import gc
import torch
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
import pandas as pd
import numpy as np
from utility_functions import set_seeds, calc_all_metrics
from models import NodeLevelGNN

def training(config,n_data_loader,runmode,tstr, variables, PATH_DICT, PATH_TENSORBOARD_SAVE=None, PATH_CSV_LOGGER=None, PATH_MODEL_CHECKPOINT=None,rand_s=42, FULL_NAME=None):
  
  
  label_ratio = round((1-tstr),4)
  
  DIR_NAME =  f"/label-ratio-{label_ratio}"
  if FULL_NAME is not None:
    FILE_NAME             = FULL_NAME + f"-label-ratio-{label_ratio}-rs-{rand_s}" 
  if PATH_TENSORBOARD_SAVE is not None:
    PATH_TENSORBOARD_SAVE = PATH_TENSORBOARD_SAVE + DIR_NAME 
  if PATH_CSV_LOGGER is not None:
    PATH_CSV_LOGGER       = PATH_CSV_LOGGER + FULL_NAME + DIR_NAME 
  if PATH_MODEL_CHECKPOINT is not None:
    PATH_MODEL_CHECKPOINT = PATH_MODEL_CHECKPOINT + FULL_NAME + DIR_NAME 
  
  
  if runmode == "tuning":    
    set_seeds(rand_s)
    
    hyperparameters_str = "-ch-" + str(config["c_hidden"]) + "-lyr-" + str(config["num_layers"]) + "-lr-" + str(config["learnr"]) + "-dpr-"+ str(config["dp_rate"]) + "-wd-" + str(config["weight_decay"])
    PATH_MODEL_CHECKPOINT = "/content/tuning_checkpoints/" + FULL_NAME + hyperparameters_str
    FILE_NAME = FULL_NAME + hyperparameters_str
    
    model_checkpoint=pl.callbacks.ModelCheckpoint(dirpath = PATH_MODEL_CHECKPOINT,
                                                  filename = FILE_NAME,
                                                  monitor = "val_acc",
                                                  save_last = False,
                                                  mode="max",
                                                  every_n_epochs=1
                                                  )
        
    trainer = pl.Trainer(
                        accelerator="auto",
                        max_epochs=variables["max_epoch"],
                        precision=16,
                        callbacks=[model_checkpoint],
                        num_sanity_val_steps=0,
                        log_every_n_steps=1
                        )


    model =NodeLevelGNN(config)
    
    train_fitting_start_time = time.time()
    trainer.fit(model = model, train_dataloaders = n_data_loader)
    train_fitting_end_time = time.time()
    train_fitting_time = train_fitting_end_time - train_fitting_start_time
    
    test_step_dict = trainer.test(model=model,dataloaders=n_data_loader,ckpt_path="best")[0]
    train_acc = test_step_dict["test_step_train_acc"]
    val_acc   = test_step_dict["test_step_val_acc"]
    test_acc  = test_step_dict["test_step_test_acc"]
    
    all_predictions = model.all_predictions.argmax(dim=-1).detach().cpu().numpy()
    truth_values    = model.all_truth_values.detach().cpu().numpy()
    test_mask_array = n_data_loader.dataset[0].test_mask.detach().cpu().numpy()
    val_mask_array  = n_data_loader.dataset[0].val_mask.detach().cpu().numpy()
    
  
    metrics_dict_for_val   = calc_all_metrics(truth_values[val_mask_array],  all_predictions[val_mask_array])
    metrics_dict_for_test  = calc_all_metrics(truth_values[test_mask_array], all_predictions[test_mask_array])

    val_precision_macro,     val_recall_macro,    val_f1_macro,_       = metrics_dict_for_val["macro_results"]
    val_precision_micro,     val_recall_micro,    val_f1_micro,_       = metrics_dict_for_val["micro_results"]
    val_precision_weighted,  val_recall_weighted, val_f1_weighted,_    = metrics_dict_for_val["weighted_results"]

    test_precision_macro,    test_recall_macro,    test_f1_macro,_     = metrics_dict_for_test["macro_results"]
    test_precision_micro,    test_recall_micro,    test_f1_micro,_     = metrics_dict_for_test["micro_results"]
    test_precision_weighted, test_recall_weighted, test_f1_weighted,_  = metrics_dict_for_test["weighted_results"]

    result = {
              "train_acc"               : train_acc,
              "val_acc"                 : val_acc,
              "test_acc"                : test_acc,
              "val-precision-macro"     : val_precision_macro,
              "val-precision-micro"     : val_precision_micro,
              "val-precision-weighted"  : val_precision_weighted,
              "val-recall-macro"        : val_recall_macro,
              "val-recall-micro"        : val_recall_micro,
              "val-recall-weighted"     : val_recall_weighted,
              "val-f1-macro"            : val_f1_macro,
              "val-f1-micro"            : val_f1_micro,
              "val-f1-weighted"         : val_f1_weighted,
              "test-precision-macro"    : test_precision_macro,
              "test-precision-micro"    : test_precision_micro,
              "test-precision-weighted" : test_precision_weighted,
              "test-recall-macro"       : test_recall_macro,
              "test-recall-micro"       : test_recall_micro,
              "test-recall-weighted"    : test_recall_weighted,
              "test-f1-macro"           : test_f1_macro,
              "test-f1-micro"           : test_f1_micro,
              "test-f1-weighted"        : test_f1_weighted,
              "train_fitting_time"      : train_fitting_time
    }
    

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result

  elif runmode=="training":
    set_seeds(rand_s)
    

    tensorboard_logger = TensorBoardLogger(PATH_TENSORBOARD_SAVE, name=FILE_NAME)
    csv_logger         = CSVLogger(PATH_CSV_LOGGER, name = FILE_NAME, flush_logs_every_n_steps = 1)
    
    model_checkpoint=pl.callbacks.ModelCheckpoint(dirpath = PATH_MODEL_CHECKPOINT,
                                                  filename = FILE_NAME,
                                                  monitor = "val_acc",
                                                  save_last = False,
                                                  mode="max",
                                                  every_n_epochs=1
                                                  )
    
    trainer = pl.Trainer(
                        accelerator="auto",
                        max_epochs=variables["max_epoch"],
                        precision=16,
                        callbacks=[model_checkpoint],
                        num_sanity_val_steps=0,
                        logger=[csv_logger, tensorboard_logger],
                        log_every_n_steps=1
                        )
    
    model =NodeLevelGNN(config)
    
    train_fitting_start_time = time.time()
    trainer.fit(model = model, train_dataloaders = n_data_loader)
    train_fitting_end_time = time.time()
    train_fitting_time = train_fitting_end_time - train_fitting_start_time
    
    test_step_dict = trainer.test(model=model,dataloaders=n_data_loader,ckpt_path="best")[0]
    train_acc = test_step_dict["test_step_train_acc"]
    val_acc   = test_step_dict["test_step_val_acc"]
    test_acc  = test_step_dict["test_step_test_acc"]
    
    all_predictions = model.all_predictions.argmax(dim=-1).detach().cpu().numpy()
    truth_values    = model.all_truth_values.detach().cpu().numpy()
    
    train_mask_array = n_data_loader.dataset[0].train_mask.detach().cpu().numpy()
    val_mask_array = n_data_loader.dataset[0].val_mask.detach().cpu().numpy()
    test_mask_array = n_data_loader.dataset[0].test_mask.detach().cpu().numpy()
    
    
    classif_report_train = sklearn.metrics.classification_report(y_true = truth_values[train_mask_array],
                                                           y_pred = all_predictions[train_mask_array],
                                                           zero_division=0,
                                                           output_dict=True)
                                                           
    classif_report_val = sklearn.metrics.classification_report(y_true = truth_values[val_mask_array],
                                                           y_pred = all_predictions[val_mask_array],
                                                           zero_division=0,
                                                           output_dict=True)                                                       
    
    classif_report_test = sklearn.metrics.classification_report(y_true = truth_values[test_mask_array],
                                                           y_pred = all_predictions[test_mask_array],
                                                           zero_division=0,
                                                           output_dict=True)
    
    
    classification_df_train = pd.DataFrame(classif_report_train)
    classification_df_val   = pd.DataFrame(classif_report_val)
    classification_df_test  = pd.DataFrame(classif_report_test)
    
    
    metrics_dict_for_train = calc_all_metrics(truth_values[train_mask_array],  all_predictions[train_mask_array])
    metrics_dict_for_val   = calc_all_metrics(truth_values[val_mask_array],  all_predictions[val_mask_array])
    metrics_dict_for_test  = calc_all_metrics(truth_values[test_mask_array], all_predictions[test_mask_array])

    train_precision_micro,   train_recall_micro,  train_f1_micro,_     = metrics_dict_for_train["micro_results"]
    val_precision_micro,     val_recall_micro,     val_f1_micro,_      = metrics_dict_for_val["micro_results"]
    test_precision_micro,    test_recall_micro,    test_f1_micro,_     = metrics_dict_for_test["micro_results"]
    
    classification_df_train["micro avg"] = [train_precision_micro, train_recall_micro, train_f1_micro, 0.0]
    classification_df_val["micro avg"]   = [val_precision_micro, val_recall_micro, val_f1_micro, 0.0]
    classification_df_test["micro avg"]  = [test_precision_micro, test_recall_micro, test_f1_micro, 0.0] 
    
    
    result = {
            "train_acc"               : train_acc,
            "val_acc"                 : val_acc,
            "test_acc"                : test_acc,
            "train_fitting_time"      : train_fitting_time,
            "classification_df_train" : classification_df_train,
            "classification_df_val"   : classification_df_val,
            "classification_df_test"  : classification_df_test
    }
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
    
  elif runmode=="thresholdvalue-searching":
    set_seeds(rand_s)


    model_checkpoint=pl.callbacks.ModelCheckpoint(dirpath = PATH_MODEL_CHECKPOINT,
                                                  filename = FILE_NAME,
                                                  monitor = "val_acc",
                                                  save_last = False,
                                                  mode="max",
                                                  every_n_epochs=1
                                                  )
    
        
    trainer = pl.Trainer(
                        accelerator="auto",
                        max_epochs=variables["max_epoch"],
                        precision=16,
                        callbacks=[model_checkpoint],
                        num_sanity_val_steps=0,
                        log_every_n_steps=1
                        )
    
    model =NodeLevelGNN(config)
    
    train_fitting_start_time = time.time()
    trainer.fit(model = model, train_dataloaders = n_data_loader)
    train_fitting_end_time = time.time()
    train_fitting_time = train_fitting_end_time - train_fitting_start_time
    
    test_step_dict = trainer.test(model=model,dataloaders=n_data_loader,ckpt_path="best")[0]
    
    train_acc = test_step_dict["test_step_train_acc"]
    val_acc   = test_step_dict["test_step_val_acc"]
    test_acc  = test_step_dict["test_step_test_acc"]
    
    
    all_predictions = model.all_predictions.argmax(dim=-1).detach().cpu().numpy()
    truth_values    = model.all_truth_values.detach().cpu().numpy()
    val_mask_array  = n_data_loader.dataset[0].val_mask.detach().cpu().numpy()
    
    metrics_dict_for_val   = calc_all_metrics(truth_labels = truth_values[val_mask_array], predicted_labels=all_predictions[val_mask_array])
    _,_, val_f1_macro,_    = metrics_dict_for_val["macro_results"]
    _,_, val_f1_micro,_    = metrics_dict_for_val["micro_results"]
    _,_, val_f1_weighted,_ = metrics_dict_for_val["weighted_results"]
    
    result = {
            "train_acc"          : test_step_dict["test_step_train_acc"],
            "val_acc"            : test_step_dict["test_step_val_acc"],
            "test_acc"           : test_step_dict["test_step_test_acc"],
            "val-f1-macro"       : val_f1_macro,
            "val-f1-micro"       : val_f1_micro,
            "val-f1-weighted"    : val_f1_weighted,
            "train_fitting_time" : train_fitting_time
    }
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result