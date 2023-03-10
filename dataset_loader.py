import torch
from torch_geometric.data import Data
from preprocessor import *

def load_dataset(name):
    if name == "ogb_mag":
        # load dataset to datasets
        from torch_geometric.datasets import OGB_MAG
        dataset = OGB_MAG(root='./datasets/ogb_mag', preprocess='metapath2vec')[0]
        dataset = Data(x=dataset.x_dict['paper'],
                       edge_index=dataset.edge_index_dict[('paper', 'cites', 'paper')],
                       y=dataset.y_dict['paper'],
                       train_mask = dataset.train_mask_dict['paper'],
                       test_mask = dataset.test_mask_dict['paper'])
        # # add some transformations
        # dataset = add_self_edges(dataset)
        # dataset = make_undirected(dataset)
        # get number of classes
        num_classes = torch.unique(dataset['y']).size(dim=0)
    elif name == "reddit":
        # load dataset to datasets
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root='./datasets/reddit')[0]
        # # add some transformations
        # dataset = add_self_edges(dataset)
        # dataset = make_undirected(dataset)
        # get number of classes
        num_classes = torch.unique(dataset['y']).size(dim=0)
    return dataset, num_classes