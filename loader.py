import torch
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from torch_geometric.loader import NeighborLoader
from configs.config import config

class SubgraphLoader():
  def __init__(self, dataset, batch_size, K, r):
    self.loader = NeighborLoader(dataset, 
                                 num_neighbors=[dataset.x.size(dim=0)]*r, 
                                 batch_size=batch_size, 
                                 shuffle=True)
    self.K = K
    self.r = r
    self.batch_size = batch_size
  

  def sample_batch(self):
    batch = next(iter(self.loader)).to(config.device)
    x, y = batch.x, batch.y
    edge_index = batch.edge_index
    # get degrees
    A = get_adjacency_matrix(edge_index, x.size(dim=0))
    eps = 0.0000001
    out_degrees = torch.sparse.sum(A, dim=1).to_dense() + eps
    # sample in edges
    p = self.K / (2*out_degrees[edge_index[0,:]])
    mask = torch.rand(p.size(dim=0)).to(config.device) < p
    sampled_edge_index = edge_index[:, mask]
    # check that no nodes have more in-degree than K
    A = get_adjacency_matrix(sampled_edge_index, x.size(dim=0))
    out_degrees = torch.sparse.sum(A, dim=1).to_dense()
    mask = out_degrees <= self.K
    mask = mask.to(config.device)
    # filter x, y, and edge_index according to nodes with out-degree 
    # greater than K
    x, y = x[mask], y[mask]
    edge_index = filter_edge_index(sampled_edge_index, mask)

    # redo subgraph construction
    batch_size = torch.count_nonzero(mask[:self.batch_size])
    sampled_indices = torch.arange(batch_size)
    sample_mask = index_to_mask(sampled_indices, size=x.size(dim=0)).to(config.device)
    for hop in range(self.r):
      edge_mask = sample_mask[edge_index[1, :].to(torch.long)]
      sample_edge_index = edge_index[:, edge_mask]
      sample_mask[edge_index[0, :].to(torch.long)] = True
    x, y = x[sample_mask], y[sample_mask]
    edge_index = filter_edge_index(edge_index, sample_mask)
    return Data(x=x, y=y, edge_index=edge_index, batch_size=batch_size)