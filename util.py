#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:49:05 2021

@author: illusionist
"""

# util.py

import datetime
import errno
import numpy as np
import os
import pickle
import random
import torch
import scipy.sparse as sp
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from torch_geometric.data import download_url, extract_zip
from csv import DictReader
from csv import reader
import numpy as np
import pandas as pd
import xlrd
import csv
import os
import csv
import xlrd
import _thread
import time
import torch as th
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader

def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0] = 10000000000
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def negsamp_incr(unique_drug_id, unique_microbe_id, unique_disease_id, drug_microbe_disease_list, n_drug_items=270, n_microbe_items=58, n_disease_items=167, n_samp=2763):
    """ Guess and check with arbitrary positivity check
    """
    neg_inds = []
    neg_drug_id=[]
    neg_microbe_id=[]
    neg_disease_id=[]
    while len(neg_inds) < n_samp:
        drug_samp = np.random.randint(0, n_drug_items)
        microbe_samp = np.random.randint(0, n_microbe_items)
        disease_samp = np.random.randint(0, n_disease_items)
        if [drug_samp, microbe_samp, disease_samp] not in drug_microbe_disease_list and [drug_samp, microbe_samp, disease_samp] not in neg_inds:
          neg_drug_id.append(drug_samp)
          neg_microbe_id.append(microbe_samp)
          neg_disease_id.append(disease_samp)
          neg_inds.append([drug_samp, microbe_samp, disease_samp])
    return neg_drug_id, neg_microbe_id, neg_disease_id, neg_inds

# The configuration below is from the paper.
default_configure = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [32, 8, 2],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout":  0.6,  # result is sensitive to dropout
        "lr":1e-5,
        "weight_decay": 0.001,
        "num_of_epochs": 100,
        "patience": 100,
        'batch_size': 32
    }


def setup(args):
    args.update(default_configure)
    args['hetero']=True
    args['dataset'] = 'DrugBank' if args['hetero'] else 'DrugBank'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args
          
def negsamp_incr(unique_drug_id, unique_microbe_id, unique_disease_id, drug_microbe_disease_list, n_drug_items=270, n_microbe_items=58, n_disease_items=167, n_samp=2763):
    """ Guess and check with arbitrary positivity check
    """
    neg_inds = []
    neg_drug_id=[]
    neg_microbe_id=[]
    neg_disease_id=[]
    while len(neg_inds) < n_samp:
        drug_samp = np.random.randint(0, n_drug_items)
        target_samp = np.random.randint(0, n_microbe_items)
        disease_samp = np.random.randint(0, n_disease_items)
        if [drug_samp, target_samp, disease_samp] not in drug_microbe_disease_list and [drug_samp, target_samp, disease_samp] not in neg_inds:
          neg_drug_id.append(drug_samp)
          neg_microbe_id.append(target_samp)
          neg_disease_id.append(disease_samp)
          neg_inds.append([drug_samp, target_samp, disease_samp])
    return neg_drug_id, neg_microbe_id, neg_disease_id, neg_inds
            
def load_drug_data(device):

    triplet_path = '/content/drive/MyDrive/HTN-main_Code/HTN-main/adj_del_4mic_myid.txt'
    triplet_df = pd.read_csv(triplet_path, delimiter='\t', header=None)
    triplet_df.columns = ['Drugs', 'Microbes', 'Diseases', 'Connection']
    print(triplet_df)

    
    #drug_target_interactions_column_names = list(drug_target_interactions.columns)

    #drug_disease_column_names = list(drug_disease_interactions.columns)

    # Create a dictionary of drugs
    unique_drug_id = triplet_df["Drugs"].unique()
    unique_drug_id = pd.DataFrame(data={
        'drugId': unique_drug_id,
        'mappedID': pd.RangeIndex(len(unique_drug_id)),
    })

    # Create a dictionary of microbes
    unique_microbe_id = triplet_df["Microbes"].unique()
    unique_microbe_id = pd.DataFrame(data={
        'microbeId': unique_microbe_id,
        'mappedID': pd.RangeIndex(len(unique_microbe_id)),
    })
   
    
    # Create a dictionary of diseases
    unique_disease_id = triplet_df["Diseases"].unique()
    unique_disease_id = pd.DataFrame(data={
        'diseaseId': unique_disease_id,
        'mappedID': pd.RangeIndex(len(unique_disease_id)),
    })

    print("unique_drug_id", unique_drug_id.shape)
    print(unique_drug_id)

    print("unique_microbe_id", unique_microbe_id.shape)
    print(unique_microbe_id)

    print("unique_disease_id", unique_disease_id.shape)
    print(unique_disease_id)

    #drug_microbe_disease_interactions = pd.merge(drug_target_interactions, drug_disease_interactions, how='inner', left_on = 'ID', right_on = 'Drug id')
    #drug_microbe_disease_interactions = drug_microbe_disease_interactions.drop("Drug id",axis=1)
    
    #print("drug_microbe_disease_interactions", drug_microbe_disease_interactions.shape)
    #print(drug_microbe_disease_interactions)

    #drug_id = pd.merge(drug_microbe_disease_interactions['Drugs'], unique_drug_id,
   #                         left_on='Drugs', right_on='drugId', how='left')
    #drug_id = torch.from_numpy(drug_id['mappedID'].values)

   # microbe_id = pd.merge(drug_microbe_disease_interactions['Microbes'], unique_microbe_id,
   #                             left_on='Microbes', right_on='microbeId', how='left')
    #microbe_id = torch.from_numpy(microbe_id['mappedID'].values)

    #disease_id = pd.merge(drug_microbe_disease_interactions['Diseases'], unique_disease_id,
    #                           left_on='Diseases', right_on='diseaseId', how='left')
    #disease_id = torch.from_numpy(disease_id['mappedID'].values)
    # Get mapped IDs directly from triplet_df
    drug_id = pd.merge(triplet_df, unique_drug_id, left_on='Drugs', right_on='drugId', how='left')['mappedID']
    drug_id = torch.from_numpy(drug_id.values).to(device)

    microbe_id = pd.merge(triplet_df, unique_microbe_id, left_on='Microbes', right_on='microbeId', how='left')['mappedID']
    microbe_id = torch.from_numpy(microbe_id.values).to(device)

    disease_id = pd.merge(triplet_df, unique_disease_id, left_on='Diseases', right_on='diseaseId', how='left')['mappedID']
    disease_id = torch.from_numpy(disease_id.values).to(device)
    # Concatenate both edge_index matrices
    edge_index = torch.column_stack([drug_id, microbe_id, disease_id])
    edge_index=edge_index.to(device)

    labels_np = edge_index.numpy()
    if np.isnan(labels_np).any() or np.isinf(labels_np).any():
      print("Warning: Labels contain NaN or infinite values.")

    #drug_feature = pd.merge(unique_drug_id['drugId'], triplet_df,
    #                        left_on='drugId', right_on='ID', how='left')
    #drug_feature = drug_feature.drop(['ID','SMILES'], axis=1)
    #drug_feature.rename( columns={'Unnamed: 0':'mappedID'}, inplace=True )

    #drug_feat = np.zeros([270,32],dtype=int)
    #for i in drug_feature.index:
    #  row = drug_feature.loc[i,'mappedID']
    #  for j in range(32):
    #    drug_feat[row][j] = drug_feature.loc[i,'EPFS_SMILES'][j+1]
    
    #drug_features = torch.tensor(drug_feat, dtype=torch.float)
    drug_feat = np.random.randint(2, size=(270, 32))
    drug_features = torch.tensor(drug_feat, dtype=torch.float)
    #microbe_feature = pd.merge(unique_microbe_id['microbeId'], triplet_df,
    #                            left_on='targetId', right_on='ID', how='left')
    #microbe_feature = microbe_feature.drop(['ID','ACS'], axis=1)
    #microbe_feature.rename( columns={'Unnamed: 0':'mappedID'}, inplace=True )

    #microbe_feat = np.zeros([58,32],dtype=int)
    #for i in microbe_feature.index:
    #  row = microbe_feature.loc[i,'mappedID']
    #  for j in range(32):
    #    microbe_feat[row][j] = microbe_feature.loc[i,'EPFS_ACS'][j+1]
    #microbe_features = torch.tensor(target_feat, dtype=torch.float)
    microbe_feat = np.random.randint(2, size=(58, 32))
    microbe_features = torch.tensor(microbe_feat, dtype=torch.float)
#change size to 167 due to new data having 167 diseases
    disease_feat = np.random.randint(2, size=(167, 32))
    disease_features = torch.tensor(disease_feat, dtype=torch.float)
    print(drug_feat)
    #print('Disese feat',disease_feat)
    #print(microbe_feat) 

    # Concatenate node features
    all_node_features = torch.cat([drug_features, microbe_features, disease_features], dim=0)
    
    #Normalize the features (helps with training)
    all_node_features = normalize_features(all_node_features)
    all_node_features = torch.from_numpy(all_node_features).to(device)

    # Create the heterogeneous graph
    hetero_graph = Data(x=all_node_features, edge_index=edge_index).to(device)
    hetero_graph = hetero_graph.to(device)

    input_data_np = all_node_features.numpy()
    if np.isnan(input_data_np).any() or np.isinf(input_data_np).any():
      print("Warning: Input data contains NaN or infinite values.")

    #print('Datatype:', node_features_csr.dtype)

    #node_features_csr = torch.from_numpy(node_features_csr).float().to(device)

    drug_id_list = drug_id.tolist()

    microbe_id_list = microbe_id.tolist()

    disease_id_list = disease_id.tolist()

    drug_microbe_disease_list =[]

    for i in range(2763):
      lst=[]
      lst.append(drug_id_list[i])
      lst.append(microbe_id_list[i])
      lst.append(disease_id_list[i])
      drug_microbe_disease_list.append(lst)

    #print('Datatype:', node_features_csr.dtype)

    #node_features_csr = torch.from_numpy(node_features_csr).float().to(device)

    
    neg_drug_id, neg_microbe_id, neg_disease_id, neg_drug_microbe_disease_list = negsamp_incr(unique_drug_id, unique_microbe_id, unique_disease_id, drug_microbe_disease_list)

    
    #graph_data = ()


    return drug_microbe_disease_list, neg_drug_microbe_disease_list, all_node_features, hetero_graph, drug_id, microbe_id, disease_id, unique_drug_id, unique_microbe_id, unique_disease_id


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))