import os
import torch
import pandas as pd
import gc
from csv import writer
from configs.config import config, experiments, iterations, logging
from torch_geometric.loader import NeighborLoader
from model import *
from train_test import *
from analysis import *
from loader import *
from dataset_loader import *
from pyvacy import optim, analysis

MAX_DEGREE = 100 # this is just so that graphs can fit in GPU

# TODO: Make it specify the output file
def main():
  # TODO: Make it optional to have these prints
  print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:>0.2f} GB")
  data_file = "./data/results.csv"
  if not os.path.isfile(data_file):
    header = [*list(config), "clipping_threshold", "sigma", "alpha", "gamma", "step", "train_acc", "test_acc"]
    with open(data_file, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(header)
        f_object.close()
  # do experiments
  num_experiments = len(list(experiments.values())[0])
  for iteration in range(iterations):
    for index in range(num_experiments):
      torch.cuda.empty_cache()
      gc.collect()
      exp_config = config.copy()
      line = "Experiment " + str((index+1) + (iteration*num_experiments)) + ": ["
      # apply experiment conditions
      for key in experiments:
        exp_config[key] = experiments[key][index]
        line += key + "=" + str(experiments[key][index]) + " "
      line = line[:-1]
      line += "]"
      print(line)

      # run experiment
      # TODO: make it so that it dynamically returns the necessary metric
      if exp_config["setup"] == "original":
        run_original_experiment(exp_config, data_file, wordy=logging)
      elif exp_config["setup"] == "ours":
        run_our_experiment(exp_config, data_file, wordy=logging)
      elif exp_config["setup"] == "non-dp":
        run_non_private_experiment(exp_config, data_file, wordy=logging)


def run_non_private_experiment(config, data_file, wordy=False):
  if wordy:
    print("Running non-DP setup")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = config.copy()

  # add constants
  row["clipping_threshold"] = 0
  row["sigma"] = 0
  row["alpha"] = 0
  row["gamma"] = 0

  # prepare data
  dataset, num_classes = load_dataset(config["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)

  # setup loaders
  train_loader = NeighborLoader(dataset, 
                                num_neighbors=[MAX_DEGREE] * config["r_hop"],
                                batch_size=config["batch_size"],
                                shuffle=True,
                                input_nodes=dataset.train_mask)
  test_loader = NeighborLoader(dataset,
                               num_neighbors=[MAX_DEGREE] * config["r_hop"],
                               batch_size=config["batch_size"],
                               input_nodes=dataset.test_mask)

  # setup model
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      params=model.parameters(), 
      lr=config["lr"], 
      weight_decay=config["weight_decay"])

  # train/test
  t = 1
  max_iters = 4000
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
    print("Running original setup")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = config.copy()

  # prepare data
  dataset, num_classes = load_dataset(config["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)

  # get clipping threshold by using clipping_percentile of the gradients
  clipping_threshold = get_clipping_threshold(dataset, config) # TODO: use train_mask
  row["clipping_threshold"] = clipping_threshold

  # get sigma according to the equation in section 6
  sigma = get_sigma(config, clipping_threshold)
  row["sigma"] = sigma

  # setup loaders
  sampled_dataset = sample_edgelists(dataset, config["degree_bound"])
  train_loader = NeighborLoader(sampled_dataset, 
                                num_neighbors=[-1] * config["r_hop"],
                                batch_size=config["batch_size"],
                                input_nodes=sampled_dataset.train_mask,
                                shuffle=True)
  test_loader = NeighborLoader(dataset,
                               num_neighbors=[MAX_DEGREE] * config["r_hop"],
                               batch_size=config["batch_size"],
                               input_nodes=dataset.test_mask)
  non_priv_train_loader = NeighborLoader(dataset,
                                         num_neighbors=[MAX_DEGREE] * config["r_hop"],
                                         batch_size=config["batch_size"],
                                         input_nodes=dataset.train_mask,
                                         shuffle=True)

  # search for alpha
  alpha, gamma = search_for_alpha(dataset.train_mask.sum().item(), sigma, clipping_threshold, config)
  row["alpha"] = alpha
  row["gamma"] = gamma

  # setup model
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = get_optimizer(config, clipping_threshold, sigma, model)

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
      _, train_acc = batch_test(next(iter(non_priv_train_loader)), "TRAIN", model, loss_fn, wordy=wordy)
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
    print("Running our setup")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = config.copy()

  # prepare data
  dataset, num_classes = load_dataset(config["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)

  # get clipping threshold by using clipping_percentile of the gradients
  clipping_threshold = get_clipping_threshold(train_dataset, config)
  row["clipping_threshold"] = clipping_threshold

  # get sigma according to the equation in section 6
  sigma = get_sigma(config, clipping_threshold)
  row["sigma"] = sigma

  # setup loaders
  test_loader = NeighborLoader(dataset,
                               num_neighbors=[MAX_DEGREE] * config["r_hop"],
                               batch_size=config["batch_size"],
                               input_nodes=dataset.test_mask)
  non_priv_train_loader = NeighborLoader(dataset,
                                         num_neighbors=[MAX_DEGREE] * config["r_hop"],
                                         batch_size=config["batch_size"], 
                                         input_nodes=dataset.train_mask,
                                         shuffle=True)

  # search for alpha
  alpha, gamma = search_for_alpha(dataset.train_mask.sum().item(), sigma, clipping_threshold, config)
  row["alpha"] = alpha
  row["gamma"] = gamma

  # setup model
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  optimizer = get_optimizer(config, clipping_threshold, sigma, model)

  # train/test
  t = 1
  while True:
    sampled_dataset = sample_edgelists(dataset, config["degree_bound"])
    train_loader = NeighborLoader(sampled_dataset, 
                                  num_neighbors=[-1] * config["r_hop"],
                                  batch_size=config["batch_size"],
                                  input_nodes=sampled_dataset.train_mask,
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
      _, train_acc = batch_test(next(iter(non_priv_train_loader)), "TRAIN", model, loss_fn, wordy=wordy)
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


def search_for_alpha(n, sigma, clipping_threshold, config):
  alpha, gamma = 1.01, np.inf
  for alpha_ in np.linspace(1.01, 40, num=200):
    gamma_ = get_gamma(n, config["batch_size"], clipping_threshold, sigma, config["r_hop"], 
                      config["degree_bound"], alpha_, config["delta"])

    if (get_epsilon(gamma_, 1, alpha_, config["delta"]) < get_epsilon(gamma, 1, alpha, config["delta"])):
      alpha = alpha_
      gamma = gamma_
  return alpha, gamma


def get_clipping_threshold(dataset, config):
  model = GNN(config["encoder_dimensions"], config["decoder_dimensions"], config["r_hop"], config["dropout"]).to(config["device"])
  loss_fn = nn.CrossEntropyLoss()
  sampled_dataset = sample_edgelists(dataset, config["degree_bound"])
  dataloader = NeighborLoader(sampled_dataset, 
                              num_neighbors=[-1] * config["r_hop"],
                              batch_size=config["batch_size"],
                              input_nodes=sampled_dataset.train_mask,
                              shuffle=True)
  clipping_threshold = config["clipping_multiplier"] * get_gradient_percentile(model, loss_fn, dataloader, config["clipping_percentile"])
  torch.cuda.empty_cache()
  gc.collect()
  return clipping_threshold


def get_sigma(config, clipping_threshold):
  return config["noise_multiplier"] * 2 * clipping_threshold * get_N(config["degree_bound"], config["r_hop"])


def get_optimizer(config, clipping_threshold, sigma, model):
  optimizer = None
  if config["optimizer"] == "DPSGD":
    optimizer = optim.DPSGD(l2_norm_clip=clipping_threshold, noise_multiplier=sigma, batch_size=config["batch_size"],
                            params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  elif config["optimizer"] == "DPAdam":
    optimizer = optim.DPAdam(l2_norm_clip=clipping_threshold, noise_multiplier=sigma, batch_size=config["batch_size"],
                             params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  elif config["optimizer"] == "DPAdamFixed":
    from dp_nlp.adam_corr import AdamCorr
    # TODO: What is eps_root?
    optimizer = AdamCorr(dp_l2_norm_clip=clipping_threshold, dp_noise_multiplier=sigma, dp_batch_size=config["batch_size"],
                         eps_root=1e-8, params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  elif config["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  return optimizer


if __name__ == '__main__':
  main()
