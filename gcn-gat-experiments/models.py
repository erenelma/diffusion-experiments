import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import sys


import pytorch_lightning as pl

import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import torch_geometric.loader as geom_loader
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from pytorch_lightning.loggers import CSVLogger





gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
  }


class GNNModel(nn.Module):
  
  def __init__(self, config, add_self_loops=None, normalize=None, cached=None):
    super().__init__()
    self.c_in=config["c_in"]
    self.c_hidden=config["c_hidden"]
    self.c_out=config["c_out"]
    self.num_layers=config["num_layers"]
    self.layer_name=config["layer_name"]
    self.dp_rate=config["dp_rate"]
    
    gnn_layer = gnn_layer_by_name[self.layer_name]
    
    layers = []
    in_channels, out_channels = self.c_in, self.c_hidden
    for l_idx in range(self.num_layers-1):
      if self.layer_name == "GCN":
        layers += [
            gnn_layer(in_channels=in_channels, 
                      out_channels=out_channels,
                      add_self_loops=add_self_loops,
                      normalize=normalize,
                      cached=cached),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dp_rate)
        ]
      elif self.layer_name == "GAT":
        layers += [
            gnn_layer(in_channels=in_channels, 
                      out_channels=out_channels,
                      add_self_loops=add_self_loops),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dp_rate)
        ]
      in_channels = self.c_hidden
      
    if self.layer_name == "GCN":
      layers += [gnn_layer(in_channels=in_channels, 
                          out_channels=self.c_out,
                          add_self_loops=add_self_loops,
                          normalize=normalize,
                          cached=cached)
                 ]
                          
    elif self.layer_name == "GAT":
      layers += [gnn_layer(in_channels=in_channels, 
                            out_channels=self.c_out)
                 ]
    self.layers = nn.ModuleList(layers)
  
  def forward(self, x, edge_index, edge_weight):
    for l in self.layers:
      if isinstance(l, geom_nn.MessagePassing):
        if isinstance(l, geom_nn.GATConv):
          x = l(x=x, edge_index = edge_index, edge_attr = edge_weight)   
        if isinstance(l, geom_nn.GCNConv):
          x = l(x=x, edge_index = edge_index, edge_weight = edge_weight)
      else:
        x = l(x)
    return x

class NodeLevelGNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.weight_decayx=config["weight_decay"]
        self.learnr=config["learnr"]
        self.model_name=config["model_name"]
        self.layer_name=config["layer_name"]
        if self.layer_name == "GCN":
          self.model = GNNModel(config , add_self_loops=True, normalize=True, cached=False)
        elif self.layer_name == "GAT":
          self.model = GNNModel(config)
        self.loss_module = nn.CrossEntropyLoss()
        self.train_acc_list=[]
        self.val_acc_list=[]
        self.test_acc_list=[]
        self.train_loss_list=[]
        self.val_loss_list=[]
        self.all_predictions=None
        self.all_truth_values=None
        self.best_epoch = None
        self.save_hyperparameters()
    def forward(self, data):
      x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
      if data.edge_index is not None and data.edge_weight is not None:
        x = self.model(x, edge_index , edge_weight)
      elif data.edge_index is None and data.edge_weight is None:
        x = self.model(x)
      else:
        sys.exit(0)
      return x
        
    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.learnr, weight_decay=self.weight_decayx)
      pla_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min' , factor=0.9, patience=100)
      con_sch = {
        "scheduler": pla_scheduler ,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "val_loss",
        "strict":True,
        "name":None, 
      }
      return [optimizer] , [con_sch]

    def training_step(self, batch, batch_idx):
      all_preds = self.forward(batch)
      train_loss = self.loss_module(all_preds[batch.train_mask], batch.y[batch.train_mask])
      self.train_loss_list.append(train_loss.item())
      self.log('train_loss', train_loss)
      return {"loss": train_loss,
              "training_step_batch": batch}

    def test_step(self, batch, batch_idx):
      all_predictions = self.forward(batch)
      self.all_predictions = all_predictions
      self.all_truth_values = batch.y
      
      train_acc = (all_predictions[batch.train_mask].argmax(dim=-1) == batch.y[batch.train_mask]).sum().float() / batch.train_mask.sum()
      val_acc   = (all_predictions[batch.val_mask].argmax(dim=-1)   == batch.y[batch.val_mask]).sum().float()   / batch.val_mask.sum()
      test_acc  = (all_predictions[batch.test_mask].argmax(dim=-1)  == batch.y[batch.test_mask]).sum().float() / batch.test_mask.sum()
      
      self.best_epoch = self.current_epoch
      self.log("test_step_train_acc", train_acc)
      self.log("test_step_val_acc", val_acc)
      self.log("test_step_test_acc", test_acc)
      self.log("best_epoch", float(self.current_epoch))
    
    def training_epoch_end(self, training_step_outputs):
      batch = training_step_outputs[0]["training_step_batch"]
      with torch.no_grad(), torch.autocast(device_type="cuda"):
        self.model.eval()
        all_preds = self.forward(batch)
        val_loss  = self.loss_module(all_preds[batch.val_mask], batch.y[batch.val_mask])
        train_acc = (all_preds[batch.train_mask].argmax(dim=-1) == batch.y[batch.train_mask]).sum().float() / batch.train_mask.sum()
        val_acc   = (all_preds[batch.val_mask].argmax(dim=-1)   == batch.y[batch.val_mask]).sum().float()   / batch.val_mask.sum()
        test_acc  = (all_preds[batch.test_mask].argmax(dim=-1)  == batch.y[batch.test_mask]).sum().float()  / batch.test_mask.sum()
        self.log("train_acc", train_acc)
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        self.log("test_acc", test_acc)
        self.train_acc_list.append(train_acc.item())
        self.val_acc_list.append(val_acc.item())
        self.test_acc_list.append(test_acc.item())
        self.val_loss_list.append(val_loss.item())