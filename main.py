#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:58:38 2021

@author: illusionist
"""

# main.py
import argparse
from model_hetero import HTN, HTNLayer, BioEncoder
from util import EarlyStopping, load_drug_data
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import argparse
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score

"""

def hit_at_n(predictions, labels, n):
    
    #Compute the Hit@N metric.

    #:param predictions: Predicted scores, numpy array of shape (num_samples, num_classes).
    #:param labels: True labels, numpy array of shape (num_samples, num_classes).
    #:param n: The cutoff for computing Hit@N.

    #:return: Hit@N score.

    num_samples = predictions.shape[0]
    num_negative_samples = 100  # Number of negative samples per test triplet

    # Get the indices of the top-N predictions for each sample
    top_n_preds = np.argsort(predictions, axis=1)[:, -n:]

    hit = 0
    for i in range(num_samples):
        sample_labels = np.concatenate((labels[i], np.random.choice(labels.shape[1], size=num_negative_samples)))
        if np.any(sample_labels[top_n_preds[i]]):
            hit += 1

    return hit / num_samples


def dcg_at_k(y_true, y_score, n):
    
    Compute DCG@k for predictions.

    :param y_true: True values, numpy array of bools.
    :param y_score: Predicted scores, numpy array of floats.
    :param k: The cutoff for computing DCG.

    :return score: DCG@k score
    
    # First, we sort the true values by the predicted score (descending order)
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:n])

    # Compute the DCG@k: sum of true values / log2(rank + 1)
    gain = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return np.sum(gain / discounts)
"""

def hit_at_n(predictions, labels, n):
    num_samples = predictions.shape[0]
    num_negative_samples = 100  # Number of negative samples per test triplet

    # Ensure labels and predictions have compatible shapes
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, axis=1)

    # Initialize an array to store the hit results
    hit_results = np.zeros(num_samples)

    for i in range(num_samples):
        sample_labels = np.concatenate((labels[i], np.random.choice(labels.shape[1], size=num_negative_samples)))
        if np.any(sample_labels):
            # Sort the predicted scores in descending order
            sorted_indices = np.argsort(predictions[i])[::-1]
            # Get the top-n predictions
            top_n_predictions = sorted_indices[:n]
            # Check if there's any intersection between top_n_predictions and relevant items
            if np.any(sample_labels[top_n_predictions]):
                hit_results[i] = 1

    # Calculate the overall hit@n score
    hit_score = np.mean(hit_results)

    return hit_score


def hit_at_n(predictions, labels, n=10):
    """
    Compute the Hit@N metric.

    :param predictions: Predicted scores, numpy array of shape (num_samples, num_classes).
    :param labels: True labels, numpy array of shape (num_samples, num_classes).
    :param n: The cutoff for computing Hit@N.

    :return: Hit@N score.
    """
    num_samples = predictions.shape[0]
    num_negative_samples = 100  # Number of negative samples per test triplet

    # Get the indices of the top-N predictions for each sample
    top_n_preds = np.argsort(predictions, axis=1)[:, -n:]

    hit = 0
    for i in range(num_samples):
        sample_labels = np.concatenate((labels[i], np.random.choice(labels.shape[1], size=num_negative_samples)))
        if np.any(sample_labels[top_n_preds[i]]):
            hit += 1

    return hit / num_samples

def dcg_at_k(y_true, y_score, n):
    """
    Compute DCG@k for predictions.

    :param y_true: True values, numpy array of bools.
    :param y_score: Predicted scores, numpy array of floats.
    :param k: The cutoff for computing DCG.

    :return score: DCG@k score
    """
    # First, we sort the true values by the predicted score (descending order)
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:n])

    # Compute the DCG@k: sum of true values / log2(rank + 1)
    gain = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return np.sum(gain / discounts)

def ndcg_at_k(y_true, y_score, k):
    # Convert to numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_score = np.array(y_score, dtype=np.float64)

    # Sort the predicted scores in descending order and get the order of items
    order = np.argsort(y_score)[::-1]

    # Get the true relevance scores for the top-k items
    y_true_sorted = np.take(y_true, order[:k])

    # Calculate the gain for the top-k items
    gain = 2 ** y_true_sorted - 1

    # Calculate the discount factor for each position
    discounts = np.log2(np.arange(k) + 2)

    # Calculate the DCG (Discounted Cumulative Gain) for the top-k items
    dcg = np.sum(gain / discounts)

    # Sort the true relevance scores in descending order and get the ideal order of items
    ideal_order = np.argsort(y_true)[::-1]

    # Get the true relevance scores for the top-k items in the ideal order
    ideal_true_sorted = np.take(y_true, ideal_order[:k])

    # Calculate the ideal DCG for the top-k items
    ideal_dcg = np.sum(2 ** ideal_true_sorted - 1 / np.log2(np.arange(k) + 2))

    # Calculate NDCG
    if ideal_dcg > 0:
        ndcg = dcg / ideal_dcg
    else:
        ndcg = 0.0

    return ndcg



def remove_test_edges(hetero_graph, test_edges):
    remaining_edges = hetero_graph.edge_index.t().tolist()
    test_edges = test_edges.tolist()

    # Remove test edges from the list of remaining edges
    remaining_edges = [edge for edge in remaining_edges if edge not in test_edges]

    # Convert the remaining edges back to the edge_index format
    remaining_edges = torch.tensor(remaining_edges, dtype=torch.long).t()

    # Create a new hetero_graph with the remaining edges
    hetero_graph_no_test = Data(x=hetero_graph.x, edge_index=remaining_edges).to(hetero_graph.x.device)

    return hetero_graph_no_test


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    device = torch.device("cpu" if torch.cuda.is_available() else "cuda")  # checking whether you have a GPU

    # Load data and prepare the model
    drug_microbe_disease_list, neg_drug_microbe_disease_list, all_node_features, hetero_graph, drug_id, microbe_id, disease_id, unique_drug_id, unique_microbe_id, unique_disease_id = load_drug_data(device)

    # Combine positive and negative samples
    drug_microbe_disease_tensor = torch.tensor(drug_microbe_disease_list, dtype=torch.long).view(-1, 3).to(device)
    neg_drug_microbe_disease_tensor = torch.tensor(neg_drug_microbe_disease_list, dtype=torch.long).view(-1, 3).to(device)

    # Concatenate positive and negative samples
    combined_drug_microbe_disease_tensor = torch.cat((drug_microbe_disease_tensor, neg_drug_microbe_disease_tensor), dim=0)

    # Create labels for positive and negative samples
    positive_labels = torch.ones(drug_microbe_disease_tensor.size(0), 1)
    negative_labels = torch.zeros(neg_drug_microbe_disease_tensor.size(0), 1)

    # Concatenate labels for positive and negative samples
    combined_labels = torch.cat((positive_labels, negative_labels), dim=0)
    labels_np = combined_labels.numpy()
    if np.isnan(labels_np).any() or np.isinf(labels_np).any():
      print("Warning: Labels contain NaN or infinite values.")

    # Convert DataFrames to dictionaries for efficient mapping
    drug_id_dict = dict(zip(unique_drug_id['mappedID'], unique_drug_id['drugId']))
    microbe_id_dict = dict(zip(unique_microbe_id['mappedID'], unique_microbe_id['microbeId']))
    disease_id_dict = dict(zip(unique_disease_id['mappedID'], unique_disease_id['diseaseId']))

    # Split combined tensor and labels into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_drug_microbe_disease_tensor, combined_labels, test_size=0.2, random_state=42)

    # Create TensorDatasets for train and test sets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    # Instantiate the GNN model
    model = HTN(args["num_of_layers"], args["num_heads_per_layer"], args["num_features_per_layer"]).to(device)
    model_MCHNN = BioEncoder()
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    time_start = time.time()


    for epoch in range(args["num_of_epochs"]):
      model.train()
      epoch_loss = 0.0

      for batch in train_loader:
          drug_microbe_disease_batch, y_batch = batch
          drug_microbe_disease_batch, y_batch = drug_microbe_disease_batch.to(device), y_batch.to(device)

          labels_np = drug_microbe_disease_batch.numpy()
          if np.isnan(labels_np).any() or np.isinf(labels_np).any():
            print("Warning: Labels contain NaN or infinite values.")

          hetero_graph_np = hetero_graph.x.numpy()
          if np.isnan(hetero_graph_np).any() or np.isinf(hetero_graph_np).any():
            print("Warning: hetero_graph_np contain NaN or infinite values.")

          y_batch_np = y_batch.numpy()
          if np.isnan(y_batch_np).any() or np.isinf(y_batch_np).any():
            print("Warning: y_batch_np contain NaN or infinite values.")
          
          optimizer.zero_grad()

          # Forward pass
          out = model(hetero_graph.x, drug_microbe_disease_batch)

          # Get the predicted interaction values
          y_pred = model.forward_predictor(out, drug_microbe_disease_batch[:, 0], drug_microbe_disease_batch[:, 1], drug_microbe_disease_batch[:, 2])

          # Calculate loss
          loss = criterion(y_pred, y_batch)

          # Backward pass and optimization
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()

          epoch_loss += loss.item()

      # Print average loss for the current epoch 
      print(f'HeTriNet training: Epoch= {epoch + 1} | Loss= {epoch_loss/ len(X_train)}')


    # Calculate standard deviation and variance for model parameters
    model_parameters = list(model.parameters())
    all_parameters = torch.cat([param.view(-1) for param in model_parameters])
    std_deviation = torch.std(all_parameters)
    variance = torch.var(all_parameters)

    print("Standard Deviation:", std_deviation.item())
    print("Variance:", variance.item())


    
    # Print average loss for the current epoch
    #print(f'HTN Epoch: {epoch + 1}, Loss: {epoch_loss / len(X_train)}')
    torch.save(model.state_dict(), '/content/drive/MyDrive/HTN-main_Code/model.pth')
    model.eval()
    correct = 0
    total = 0
    y_true_list = []
    y_pred_list = []
    # Create lists to store predictions and ground truth labels
    predictions = []
    ground_truth = []
    pred = []
    triplets = []
    triplet_scores = []
    with torch.no_grad():
        for batch in test_loader:
            
            drug_microbe_disease_batch, labels = batch
            drug_microbe_disease_batch, labels = drug_microbe_disease_batch.to(device), labels.to(device)

            # Forward pass
            out = model(hetero_graph.x, drug_microbe_disease_batch)
            # Get the predicted interaction values
            y_pred = model.forward_predictor(out, drug_microbe_disease_batch[:, 0], drug_microbe_disease_batch[:, 1], drug_microbe_disease_batch[:, 2])
            # Apply a threshold to obtain binary predictions
            binary_predictions = (y_pred > 0.5).float()
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(y_pred.cpu().numpy())
            correct += (binary_predictions == labels).sum().item()
            total += labels.size(0)

            # Append predictions and ground truth labels
            predictions.extend(y_pred.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

            for i in range(len(drug_microbe_disease_batch)):
                triplet = drug_microbe_disease_batch[i]
                score = y_pred[i][0]

                drug_id_index, microbe_id_index, disease_id_index = triplet
                drug_name = drug_id_dict.get(drug_id_index.item(), 'Unknown Drug')
                microbe_name = microbe_id_dict.get(microbe_id_index.item(), 'Unknown Microbe')
                disease_name = disease_id_dict.get(disease_id_index.item(), 'Unknown Disease')
                if score >= 3:
                    print("triplet score", drug_name, microbe_name, disease_name, score)



    
    accuracy = correct / total
    # Convert lists to numpy arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Compute F1 Score
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))

    # Compute Recall and Precision
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))
    precision = precision_score(y_true, (y_pred > 0.5).astype(int))

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_true, y_pred)

    # Compute AUPR score
    aupr = average_precision_score(y_true, y_pred)
  
    # Print F1 Score, Recall, and Precision
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"ROC AUC: {roc_auc* 100:.2f}%")
    print(f"AUPR: {aupr* 100:.2f}%")

    # Example usage with k values 5, 10, and 15:
    k_values = [5, 10, 15]

    # y_true and y_pred are assumed to be defined as shown in the previous example

    for k in k_values:
        hit = hit_at_n(y_true, y_pred, k)
        print(f"Hit@{k}: {hit* 100:.2f}%")

        ndcg = ndcg_at_k(y_true, y_pred, k)
        print(f"NDCG@{k}: {ndcg* 100:.2f}%")
        
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    


if __name__ == '__main__':
    from util import setup

    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", type=bool, help='should test the model on the test dataset?', default=True)

    #args = vars(parser.parse_args())
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)