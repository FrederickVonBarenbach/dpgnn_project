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
  data_file = "./data/results.csv"
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
      line = "Experiment " + str((index+1) * (iteration+1)) + ": ["
      # apply experiment conditions
      for key in experiments:
        exp_config[key] = experiments[key][index]
        line += key + "=" + str(experiments[key][index]) + " "
      line = line[:-1]
      line += "]"
      print(line)

      # run experiment
      # TODO: make it so that it dynamically returns the necessary metric
      if config["setup"] == "original":
        run_original_experiment(exp_config, data_file, wordy=False)
      elif config["setup"] == "ours":
        run_our_experiment(exp_config, data_file, wordy=False)
      elif config["setup"] == "non-dp":
        run_non_private_experiment(exp_config, data_file, wordy=False)


def run_non_private_experiment(config, data_file, wordy=False):
  if wordy:
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = config.copy()

  # add constants
  row["sigma"] = 0
  row["alpha"] = 0
  row["gamma"] = 0

  # prepare data
  dataset, num_classes = load_dataset(config["dataset"])
  if hasattr(dataset, 'train_mask') and hasattr(dataset, 'test_mask'):
    train_dataset, test_dataset = apply_train_test_mask(dataset)
  else:
    train_dataset, test_dataset = train_test_split(dataset, 0.2)

  # data parameters
  n, d = train_dataset.x.size()
  n_test = test_dataset.x.size(dim=0)

  # setup loaders
  train_loader = NeighborLoader(train_dataset, 
                                num_neighbors=[n] * config["r_hop"],
                                batch_size=config["batch_size"],
                                shuffle=True)
  test_loader = NeighborLoader(test_dataset,
                               num_neighbors=[n_test] * config["r_hop"],
                               batch_size=config["batch_size"])

  # setup model
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      params=model.parameters(), 
      lr=config["lr"], 
      weight_decay=config["weight_decay"])

  # train/test
  t = 1
  max_iters = 10000
  while True:
    # train if haven't expended all of budget
    if t < max_iters:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer)
    if t % 100 == 0 or t >= max_iters:
      if wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
      _, train_acc = batch_test(next(iter(train_loader)), "TRAIN", model, loss_fn, wordy=wordy)
      _, test_acc = test(test_loader, "TEST", model, loss_fn, wordy=wordy)
      row["epsilon"] = 0
      row["step"] = t
      row["train_acc"] = train_acc
      row["test_acc"] = test_acc
      # write data to csv
      with open(data_file, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(row.values())
        f_object.close()

    if t >= max_iters:
      break
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1


def run_original_experiment(config, data_file, wordy=False):
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

  # setup loaders
  sampled_dataset = sample_edgelists(train_dataset, config["degree_bound"])
  train_loader = NeighborLoader(sampled_dataset, 
                                num_neighbors=[n] * config["r_hop"],
                                batch_size=config["batch_size"],
                                shuffle=True)
  test_loader = NeighborLoader(test_dataset,
                               num_neighbors=[n_test] * config["r_hop"],
                               batch_size=config["batch_size"])

  # search for alpha
  alpha, gamma = search_for_alpha(n, sigma, config)
  row["alpha"] = alpha
  row["gamma"] = gamma

  # setup model
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.DPAdam(
      l2_norm_clip=config["clipping_threshold"],
      noise_multiplier=sigma,
      batch_size=config["batch_size"],
      params=model.parameters(),
      lr=config["lr"],
      weight_decay=config["weight_decay"])

  # train/test
  t = 1
  while True:
    curr_epsilon = get_epsilon(gamma, t, alpha, config["delta"])
    # train if haven't expended all of budget
    if curr_epsilon < config["epsilon"]:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer)
    if t % 100 == 0 or curr_epsilon >= config["epsilon"]:
      if wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
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

    if curr_epsilon >= config["epsilon"]:
      break
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1


def run_our_experiment(config, data_file, wordy=False):
  if wordy:
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = config.copy()

  # get sigma according to the equation in section 6
  sigma = get_sigma(config)
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

  # setup loaders
  test_loader = NeighborLoader(test_dataset,
                               num_neighbors=[n_test] * config["r_hop"],
                               batch_size=config["batch_size"])

  # search for alpha
  alpha, gamma = search_for_alpha(n, sigma, config)
  row["alpha"] = alpha
  row["gamma"] = gamma

  # setup model
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.DPAdam(
      l2_norm_clip=config["clipping_threshold"],
      noise_multiplier=sigma,
      batch_size=config["batch_size"],
      params=model.parameters(),
      lr=config["lr"],
      weight_decay=config["weight_decay"])

  # train/test
  t = 1
  while True:
    sampled_dataset = sample_edgelists(train_dataset, config["degree_bound"])
    train_loader = NeighborLoader(sampled_dataset, 
                                  num_neighbors=[n] * config["r_hop"],
                                  batch_size=config["batch_size"],
                                  shuffle=True)
    curr_epsilon = get_epsilon(gamma, t, alpha, config["delta"])
    # train if haven't expended all of budget
    if curr_epsilon < config["epsilon"]:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer)
    if t % 100 == 0 or curr_epsilon >= config["epsilon"]:
      if wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
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

    if curr_epsilon >= config["epsilon"]:
      break
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1


def search_for_alpha(n, sigma, config):
  alpha, gamma = 1.01, np.inf
  for alpha_ in np.linspace(1.01, 40, num=200):
    gamma_ = get_gamma(n, config["batch_size"], config["clipping_threshold"], sigma, config["r_hop"], 
                      config["degree_bound"], alpha_, config["delta"])

    if (get_epsilon(gamma_, 1, alpha_, config["delta"]) < get_epsilon(gamma, 1, alpha, config["delta"])):
      alpha = alpha_
      gamma = gamma_
  return alpha, gamma


def get_sigma(config):
  return config["noise_multiplier"] * 2 * config["clipping_threshold"] * get_N(config["degree_bound"], config["r_hop"])


if __name__ == '__main__':
  main()
