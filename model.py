import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MLP
import torch.nn.functional as F

class GNN(torch.nn.Module):
  def __init__(self, encoder_dimensions, decoder_dimensions, r_hops, dropout):
      super().__init__()
      embedding_dimensions = encoder_dimensions[-1]
      self.gcn = nn.ModuleList()
      for i in range(r_hops):
        self.gcn.append(GCNConv(embedding_dimensions, embedding_dimensions, normalize=False))
      # for i in range(r_hops):
      #   if i == 0:
      #     self.gcn.append(GCNConv(embedding_dimensions, embedding_dimensions, normalize=False))
      #   else:
      #     self.gcn.append(GCNConv(embedding_dimensions, embedding_dimensions, normalize=False))
      self.encoder = MLP(encoder_dimensions)
      self.decoder = MLP(decoder_dimensions)
      self.dropout = dropout

  def forward(self, data):
      x, edge_index = data.x, data.edge_index
      x = self.encoder(x)
      for r in range(len(self.gcn)):
        x = self.gcn[r](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
      x = self.decoder(x)
      return F.log_softmax(x, dim=1)