import os
import torch
import pandas as pd
import gc
from csv import writer
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
  # TODO: Make it optional to have these prints
  print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:>0.2f} GB")
  data_file = "results.csv"
  if not os.path.isfile(data_file):
    header = [*list(config), "sigma", "alpha", "gamma", "step", "train_acc", "test_acc"]
    with open(data_file, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(header)
        f_object.close()
  # do experiments
  for index in range(len(list(experiments.values())[0])):
    for iteration in range(iterations):
      torch.cuda.empty_cache()
      gc.collect()
      exp_config = config.copy()
      # apply experiment conditions
      for key in experiments:
        exp_config[key] = experiments[key][index]

      # run experiment
      # TODO: make it so that it dynamically returns the necessary metric
      run_experiment(exp_config, data_file, wordy=True)

def run_experiment(config, data_file, wordy=False):
  if wordy:
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = config.copy()

  # get sigma according to the equation in section 6
  sigma = config["noise_multiplier"] * 2 * config["clipping_threshold"] * get_N(config["degree_bound"], config["r_hop"])
  row["sigma"] = sigma

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
  # original
  # sampled_dataset = sample_edgelists(train_dataset, config["degree_bound"])
  # train_loader = NeighborLoader(sampled_dataset, 
  #                               num_neighbors=[n] * config["r_hop"],
  #                               batch_size=config["batch_size"],
  #                               shuffle=True)
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
  row["alpha"] = alpha
  row["gamma"] = gamma

  # print hyper-parameters
  if wordy:
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
      lr=config["lr"],
      weight_decay=config["weight_decay"])
  # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

  # train/test
  t = 1
  while True:
    sampled_dataset = sample_edgelists(train_dataset, config["degree_bound"])
    train_loader = NeighborLoader(sampled_dataset, 
                                  num_neighbors=[n] * config["r_hop"],
                                  batch_size=config["batch_size"],
                                  shuffle=True)
    # batch = train_loader.sample_batch()
    curr_epsilon = get_epsilon(gamma, t, alpha, config["delta"])
    # train if haven't expended all of budget
    if curr_epsilon < config["epsilon"]:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer)
    if t % 100 == 0 or curr_epsilon >= config["epsilon"]:
      if wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
      # batch_test(train_loader.sample_batch(), "TRAIN", model, loss_fn)
      _, train_acc = batch_test(next(iter(train_loader)), "TRAIN", model, loss_fn, wordy=wordy)
      _, test_acc = test(test_loader, "TEST", model, loss_fn, wordy=wordy)
      if wordy:
        print(" Optimizer Achieves ({:>0.1f}, {})-DP".format(curr_epsilon, config["delta"]))
      row["epsilon"] = curr_epsilon
      row["step"] = t
      row["train_acc"] = train_acc
      row["test_acc"] = test_acc
      # write data to csv
      with open(data_file, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row.values())
        f_object.close()
      # print(" LR:", scheduler.get_last_lr()[0])
      if curr_epsilon >= config["epsilon"]:
        break
    # scheduler.step()
    del sampled_dataset, train_loader, batch
    # del batch
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1

if __name__ == '__main__':
  main()
