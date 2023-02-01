import os
import torch
import pandas as pd
from configs.config import config, experiments, iterations
from torch_geometric.loader import NeighborLoader
from model import *
from train_test import *
from analysis import *
from loader import *
from dataset_loader import *
from pyvacy import optim, analysis

# TODO: Make it specify the output file
def main():
  print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:>0.2f} GB")

  results = pd.DataFrame(columns=[*list(config), "accuracy"])
  # do experiments
  for index in range(len(list(experiments.values())[0])):
    for iteration in range(iterations):
      row = config.copy()
      exp_config = config.copy()
      # apply experiment conditions
      for key in experiments:
        exp_config[key] = experiments[key][index]

      # run experiment
      # TODO: make it so that it dynamically returns the necessary metric
      row["accuracy"] = run_experiment(exp_config)

      # add row
      row = pd.DataFrame(row, index=[iteration])
      results = pd.concat([results, row])
  # save results
  results.to_csv('results.csv', index=False)  

def run_experiment(config):
  # get sigma according to the equation in section 6
  sigma = config["noise_multiplier"] * 2 * config["clipping_threshold"] * get_N(config["degree_bound"], config["r_hop"])

  # prepare data
  dataset, num_classes = load_dataset(config["dataset"])
  if hasattr(dataset, 'train_mask') and hasattr(dataset, 'test_mask'):
    train_dataset, test_dataset = apply_train_test_mask(dataset)
  else:
    train_dataset, test_dataset = train_test_split(dataset, 0.2)


  # data parameters
  n, d = train_dataset.x.size()
  n_test = test_dataset.x.size(dim=0)

  # improvement
  # in_loader = SubgraphLoader(train_dataset, batch_size=batch_size, K=degree_bound, r=r_hop)
  test_loader = NeighborLoader(test_dataset,
                              num_neighbors=[n_test] * config["r_hop"],
                              batch_size=config["batch_size"])

  # search for alpha
  alpha, gamma = 1.01, np.inf
  for alpha_ in np.linspace(1.01, 20, num=100):
    gamma_ = get_gamma(n, config["batch_size"], config["clipping_threshold"], sigma, config["r_hop"], 
                      config["degree_bound"], alpha_, config["delta"])

    if (get_epsilon(gamma_, 1, alpha_, config["delta"]) < get_epsilon(gamma, 1, alpha, config["delta"])):
      alpha = alpha_
      gamma = gamma_

  # print hyper-parameters
  print(f"n: {n}, d: {d}, num_classes: {num_classes}")
  print(f"C: {config['clipping_threshold']}, sigma: {sigma}, gamma: {gamma}")
  print(f"epsilon (for 1 iteration): {get_epsilon(gamma, 1, alpha, config['delta'])}, delta: {config['delta']}, alpha: {alpha}")

    

  # setup model
  
  model = GNN([128, 256, 256], [256, 256, num_classes], config["r_hop"], 0.1).to(config["device"])
  # model = GNN(767, 64, [num_classes], r_hop).to(device)
  # model = GNN(165, 20, [num_classes], r_hop).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.DPAdam(
      l2_norm_clip=config["clipping_threshold"],
      noise_multiplier=sigma,
      batch_size=config["batch_size"],
      params=model.parameters(),
      lr=1e-3,
      weight_decay=1e-5)
  # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

  # Train/Test
  for t in range(3000):
    sampled_dataset = sample_edgelists(train_dataset, config["degree_bound"])
    train_loader = NeighborLoader(sampled_dataset, 
                                  num_neighbors=[n] * config["r_hop"],
                                  batch_size=config["batch_size"],
                                  shuffle=True)
    # batch = train_loader.sample_batch()
    batch = next(iter(train_loader))
    train(batch, model, loss_fn, optimizer)
    curr_epsilon = get_epsilon(gamma, t+1, alpha, config["delta"])
    if (t + 1) % 100 == 0:
      print("Training step:", t+1)
      # batch_test(train_loader.sample_batch(), "TRAIN", model, loss_fn)
      batch_test(next(iter(train_loader)), "TRAIN", model, loss_fn, wordy=False)
      batch_test(next(iter(test_loader)), "TEST", model, loss_fn, wordy=False)
      print(" Optimizer Achieves ({:>0.1f}, {})-DP".format(curr_epsilon, config["delta"]))
      # print(" LR:", scheduler.get_last_lr()[0])
    # scheduler.step()
    del sampled_dataset, train_loader, batch
    torch.cuda.empty_cache()
    if curr_epsilon >= config["epsilon"]:
      break
  return test(test_loader, "TEST", model, loss_fn)

if __name__ == '__main__':
  main()
