import torch
from torch_geometric.data import Data
from configs.config import config

# this method partitions based on nodes (so edges between splits are not used)
def train_test_split(dataset, test_ratio):
    x, y, edge_index= dataset.x, dataset.y, dataset.edge_index
    n = x.size(dim=0)
    shuffle_ordering = torch.randperm(n)
    # edges need to be remapped based on shuffle ordering
    # (i.e if nodes 3 and 4 are put in indices 5 and 2 respectively, 
    #      edge 3 -> 4 would becomes 5 -> 2)
    edge_mapping = torch.zeros(n, dtype=torch.long)
    edge_mapping[shuffle_ordering] = torch.arange(n)
    # shuffle x and y and apply to edge_index
    x = x[shuffle_ordering]
    y = y[shuffle_ordering]
    edge_index = edge_mapping[edge_index]
    # mask based on train and test split
    mask = torch.zeros(n, dtype=torch.bool)
    train_slice = int((1-test_ratio)*n)
    mask[:train_slice] = True
    # apply split
    x_train, x_test = x[mask], x[~mask]
    y_train, y_test = y[mask], y[~mask]
    # get edges for each split only if both nodes are in the respective split
    edge_index_train = edge_index[:, torch.logical_and(*mask[edge_index])]
    edge_index_test = edge_index[:, torch.logical_and(*~mask[edge_index])] - train_slice
    return Data(x=x_train, y=y_train, edge_index=edge_index_train), \
           Data(x=x_test, y=y_test, edge_index=edge_index_test)


def apply_train_test_mask(dataset):
  x, y, edge_index = dataset.x, dataset.y, dataset.edge_index
  train_mask, test_mask = dataset.train_mask, dataset.test_mask
  n, train_size, test_size = x.size(dim=0), \
                             torch.count_nonzero(train_mask).item(), \
                             torch.count_nonzero(test_mask).item()
  # edges need to be remapped to make a train and test dataset (so that the
  # edges are zero-indexed)
  edge_mapping = torch.zeros(n, dtype=torch.long)
  edge_mapping[train_mask] = torch.arange(train_size)
  edge_mapping[test_mask] = torch.arange(test_size)
  # apply mapping to edge_index
  remapped_edge_index = edge_mapping[edge_index]
  # do train/test split
  x_train, x_test = x[train_mask], x[test_mask]
  y_train, y_test = y[train_mask], y[test_mask]
  # apply to remapped edge_index since it is zero-index for both train and test
  # split, but use original edge_index for indexing the train_mask because that
  # one is designed for original dataset
  edge_index_train = remapped_edge_index[:, torch.logical_and(*train_mask[edge_index])]
  edge_index_test = remapped_edge_index[:, torch.logical_and(*test_mask[edge_index])]
  return Data(x=x_train, y=y_train, edge_index=edge_index_train), \
         Data(x=x_test, y=y_test, edge_index=edge_index_test)


def filter_edge_index(edge_index, mask):
  # remap edges 
  # (i.e. if we remove 3, then edge 2 -> 4 becomes 2 -> 3)
  node_indices = torch.arange(mask.size(dim=0)).to(config['device'])[mask]
  edge_mapping = torch.zeros(mask.size(dim=0), dtype=torch.long).to(config['device'])
  edge_mapping[node_indices] = torch.arange(node_indices.size(dim=0)).to(config['device'])
  # remove edges containing nodes removed by mask
  edge_mask = torch.logical_and(*mask[edge_index.to(torch.long)])
  return edge_mapping[edge_index[:, edge_mask]]


# make sparse adjacency matrix, A
def get_adjacency_matrix(edge_index, num_nodes):
    values = torch.ones(edge_index.size(dim=1), dtype = torch.int).to(config['device'])
    A = torch.sparse_coo_tensor(edge_index, values, 
                                (num_nodes, num_nodes), 
                                dtype=torch.float).to(config['device'])
    return A


def sample_edgelists(dataset, K):
  dataset = dataset.to(config['device'])
  x, y = dataset.x, dataset.y
  edge_index = dataset.edge_index
  A = get_adjacency_matrix(edge_index, x.size(dim=0))
  eps = 0.0000001
  out_degrees = torch.sparse.sum(A, dim=1).to_dense() + eps
  # sample out edges
  p = K / (2*out_degrees[edge_index[0,:]])
  mask = torch.rand(p.size(dim=0)).to(config['device']) < p
  sampled_edge_index = edge_index[:, mask]
  # check that no nodes have more in-degree than K
  A = get_adjacency_matrix(sampled_edge_index, x.size(dim=0))
  out_degrees = torch.sparse.sum(A, dim=1).to_dense()
  mask = out_degrees <= K
  mask = mask.to(config['device'])
  # filter x, y, and edge_index according to nodes with out-degree 
  # greater than K
  x, y = x[mask], y[mask]
  edge_index = filter_edge_index(sampled_edge_index, mask)
  return Data(x=x, y=y, edge_index=edge_index)


def add_self_edges(dataset):
  x = dataset.x
  self_edges = torch.stack((torch.arange(x.size(dim=0)), torch.arange(x.size(dim=0))))
  edge_index = torch.cat((dataset.edge_index, self_edges), dim=1)
  return Data(x=x, y=dataset.y, edge_index=edge_index)


def make_undirected(dataset):
  edge_index = torch.cat((dataset.edge_index, dataset.edge_index[[1, 0], :]), dim=1)
  return Data(x=dataset.x, y=dataset.y, edge_index=edge_index)