import torch
from torch_geometric.data import Data

# this method adds a train and test mask to dataset
def train_test_split(dataset, test_ratio):
    n = dataset.x.size(dim=0)
    shuffle_ordering = torch.randperm(n)
    # mask based on train and test split
    mask = torch.zeros(n, dtype=torch.bool)
    train_slice = int((1-test_ratio)*n)
    mask[shuffle_ordering[:train_slice]] = True
    dataset.train_mask = mask
    dataset.test_mask = ~mask


def filter_edge_index(edge_index, mask, device):
  # remap edges 
  # (i.e. if we remove 3, then edge 2 -> 4 becomes 2 -> 3)
  node_indices = torch.arange(mask.size(dim=0)).to(device)[mask]
  edge_mapping = torch.zeros(mask.size(dim=0), dtype=torch.long).to(device)
  edge_mapping[node_indices] = torch.arange(node_indices.size(dim=0)).to(device)
  # remove edges containing nodes removed by mask
  edge_mask = torch.logical_and(*mask[edge_index.to(torch.long)])
  return edge_mapping[edge_index[:, edge_mask]]


# make sparse adjacency matrix, A
def get_adjacency_matrix(edge_index, num_nodes, device):
    values = torch.ones(edge_index.size(dim=1), dtype = torch.int).to(device)
    A = torch.sparse_coo_tensor(edge_index, values, 
                                (num_nodes, num_nodes), 
                                dtype=torch.float).to(device)
    return A


def sample_edgelists(dataset, K, device):
  dataset = dataset.to(device)
  x, y = dataset.x, dataset.y
  edge_index = dataset.edge_index
  train_mask, test_mask = dataset.train_mask, dataset.test_mask
  A = get_adjacency_matrix(edge_index, x.size(dim=0), device)
  eps = 0.0000001
  out_degrees = torch.sparse.sum(A, dim=1).to_dense() + eps
  # sample out edges
  p = K / (2*out_degrees[edge_index[0,:]])
  mask = torch.rand(p.size(dim=0)).to(device) < p
  sampled_edge_index = edge_index[:, mask]
  # check that no nodes have more in-degree than K
  A = get_adjacency_matrix(sampled_edge_index, x.size(dim=0), device)
  out_degrees = torch.sparse.sum(A, dim=1).to_dense()
  mask = out_degrees <= K
  mask = mask.to(device)
  # filter x, y, and edge_index according to nodes with out-degree 
  # greater than K
  x, y = x[mask], y[mask]
  train_mask, test_mask = train_mask[mask], test_mask[mask]
  edge_index = filter_edge_index(sampled_edge_index, mask, device)
  sampled_dataset = Data(x=x, y=y, edge_index=edge_index)
  sampled_dataset.train_mask, sampled_dataset.test_mask = train_mask, test_mask
  return sampled_dataset.cpu()


def get_gradient_percentile(model, loss_fn, dataloader, percentile, device):
  # run one iteration without training
  batch = next(iter(dataloader))
  batch = batch.to(device)
  pred = model(batch)[:batch.batch_size]
  y = batch.y[:batch.batch_size]
  loss = loss_fn(pred, y)
  loss.backward()
  # store grads
  grads = []
  for param in model.parameters():
      grads.append(param.grad.view(-1))
  grads = torch.cat(grads)
  # get percentile
  clipping_threshold = torch.quantile(grads, percentile)
  return clipping_threshold.item()


def add_self_edges(dataset):
  x = dataset.x
  self_edges = torch.stack((torch.arange(x.size(dim=0)), torch.arange(x.size(dim=0))))
  edge_index = torch.cat((dataset.edge_index, self_edges), dim=1)
  return Data(x=x, y=dataset.y, edge_index=edge_index)


def make_undirected(dataset):
  edge_index = torch.cat((dataset.edge_index, dataset.edge_index[[1, 0], :]), dim=1)
  return Data(x=dataset.x, y=dataset.y, edge_index=edge_index)