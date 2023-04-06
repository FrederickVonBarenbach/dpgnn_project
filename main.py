import os
import torch
import gc
import argparse
from csv import writer
from torch_geometric.loader import NeighborLoader
from model import *
from train_test import *
from analysis import *
from dataset_loader import *
from pyvacy import optim, analysis


def run_experiment(experiment_vars, config):
  print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:>0.2f} GB")
  line = "Experiment " + str(experiment_vars["id"]) + " : ["
  # apply experiment conditions
  for key in experiment_vars:
    line += key + "=" + str(experiment_vars[key]) + ", "
  line = line[:-2]
  line += "]"
  print(line)

  # run experiment
  # TODO: make it so that it dynamically returns the necessary metric
  if config.setup == "original":
    run_original_experiment(config, experiment_vars)
  elif config.setup == "ours":
    run_our_experiment(config, experiment_vars)
  elif config.setup == "non-dp":
    run_non_private_experiment(config, experiment_vars)

  torch.cuda.empty_cache()
  gc.collect()


def run_non_private_experiment(config, experiment_vars):
  if config.wordy:
    print("Running non-DP setup")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = experiment_vars.copy()

  # add constants
  row["clipping_threshold"] = 0
  row["sigma"] = 0
  row["alpha"] = 0
  row["gamma"] = 0

  # prepare data
  dataset, num_classes = load_dataset(experiment_vars["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)

  # setup loaders
  train_loader = NeighborLoader(dataset, 
                                num_neighbors=[config.max_degree] * experiment_vars["r_hop"],
                                batch_size=config.test_batch_size,
                                input_nodes=dataset.train_mask,
                                shuffle=True)
  test_loader = NeighborLoader(dataset,
                               num_neighbors=[config.max_degree] * experiment_vars["r_hop"],
                               batch_size=config.test_batch_size,
                               input_nodes=dataset.test_mask,
                               shuffle=True)

  # setup wandb
  if config.wandb_project is not None:
    import wandb
    wandb.init(project=config.wandb_project, config=row, name="Experiment " + str(row["id"]))

  # setup model
  model = GNN(experiment_vars["encoder_dimensions"], experiment_vars["decoder_dimensions"], 
              experiment_vars["activation"], experiment_vars["r_hop"], experiment_vars["dropout"]).to(config.device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      params=model.parameters(), 
      lr=experiment_vars["lr"], 
      weight_decay=experiment_vars["weight_decay"])

  # train/test
  t = 1
  max_iters = 4000
  while True:
    # train if haven't expended all of budget
    if t < max_iters:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer, config.device)
    if t % config.test_stepsize == 0 or t >= max_iters:
      if config.wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
      _, train_acc = n_batch_test(train_loader, config.train_batches, "TRAIN", model, loss_fn, config.device, wordy=config.wordy)
      _, test_acc = n_batch_test(test_loader, config.train_batches, "TEST", model, loss_fn, config.device, wordy=config.wordy)
      row["epsilon"] = 0
      row["step"] = t
      row["train_acc"] = train_acc
      row["test_acc"] = test_acc
      # write data to csv
      if config.wandb_project is None:
        with open(config.results_path, 'a') as f_object:
          writer_object = writer(f_object)
          writer_object.writerow(row.values())
          f_object.close()
      else:
        wandb.log({"train_acc": row["train_acc"], "test_acc": row["test_acc"], "step": row["step"], "epsilon": row["epsilon"]})

    if t >= max_iters:
      break
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1


def run_original_experiment(config, experiment_vars):
  if config.wordy:
    print("Running original setup")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = experiment_vars.copy()

  # prepare data
  dataset, num_classes = load_dataset(experiment_vars["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)

  # get clipping threshold by using clipping_percentile of the gradients
  clipping_threshold = get_clipping_threshold(dataset, experiment_vars)
  row["clipping_threshold"] = clipping_threshold

  # get sigma according to the equation in section 6
  sigma = get_sigma(experiment_vars, clipping_threshold)
  row["sigma"] = sigma

  # setup loaders
  sampled_dataset = sample_edgelists(dataset, experiment_vars["degree_bound"], config.device)
  train_loader = NeighborLoader(sampled_dataset, 
                                num_neighbors=[-1] * experiment_vars["r_hop"],
                                batch_size=experiment_vars["batch_size"],
                                input_nodes=sampled_dataset.train_mask,
                                shuffle=True)

  test_loader = NeighborLoader(dataset,
                               num_neighbors=[config.max_degree] * experiment_vars["r_hop"],
                               batch_size=config.test_batch_size,
                               input_nodes=dataset.test_mask,
                               shuffle=True)

  non_priv_train_loader = NeighborLoader(dataset,
                                         num_neighbors=[config.max_degree] * experiment_vars["r_hop"],
                                         batch_size=config.test_batch_size,
                                         input_nodes=dataset.train_mask,
                                         shuffle=True)

  # search for alpha
  alpha, gamma = search_for_alpha(dataset.train_mask.sum().item(), sigma, clipping_threshold, experiment_vars)
  row["alpha"] = alpha
  row["gamma"] = gamma

  # setup wandb
  if config.wandb_project is not None:
    import wandb
    wandb.init(project=config.wandb_project, config=row, name="Experiment " + str(row["id"]))

  # setup model
  model = GNN(experiment_vars["encoder_dimensions"], experiment_vars["decoder_dimensions"], 
              experiment_vars["activation"], experiment_vars["r_hop"], experiment_vars["dropout"]).to(config.device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = get_optimizer(experiment_vars, clipping_threshold, sigma, model)

  # train/test
  t = 1
  while True:
    curr_epsilon = get_epsilon(gamma, t, alpha, experiment_vars["delta"])
    # train if haven't expended all of budget
    if curr_epsilon < experiment_vars["epsilon"]:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer, config.device)
    if t % config.test_stepsize == 0 or curr_epsilon >= experiment_vars["epsilon"]:
      if config.wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
      _, train_acc = n_batch_test(non_priv_train_loader, config.train_batches, "TRAIN", model, loss_fn, config.device, wordy=config.wordy)
      _, test_acc = n_batch_test(test_loader, config.train_batches, "TEST", model, loss_fn, config.device, wordy=config.wordy)
      if config.wordy:
        print(" Optimizer Achieves ({:>0.1f}, {})-DP".format(curr_epsilon, experiment_vars["delta"]))
      row["epsilon"] = curr_epsilon
      row["step"] = t
      row["train_acc"] = train_acc
      row["test_acc"] = test_acc
      # write data to csv
      if config.wandb_project is None:
        with open(config.results_path, 'a') as f_object:
          writer_object = writer(f_object)
          writer_object.writerow(row.values())
          f_object.close()
      else:
        wandb.log({"train_acc": row["train_acc"], "test_acc": row["test_acc"], "step": row["step"], "epsilon": row["epsilon"]})

    if curr_epsilon >= experiment_vars["epsilon"]:
      break
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1


def run_our_experiment(config, experiment_vars):
  if config.wordy:
    print("Running our setup")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")

  # make a new data row
  row = experiment_vars.copy()

  # prepare data
  dataset, num_classes = load_dataset(experiment_vars["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)

  # get clipping threshold by using clipping_percentile of the gradients
  clipping_threshold = get_clipping_threshold(dataset, experiment_vars)
  row["clipping_threshold"] = clipping_threshold

  # get sigma according to the equation in section 6
  sigma = get_sigma(experiment_vars, clipping_threshold)
  row["sigma"] = sigma

  # setup loaders
  test_loader = NeighborLoader(dataset,
                               num_neighbors=[config.max_degree] * experiment_vars["r_hop"],
                               batch_size=config.test_batch_size,
                               input_nodes=dataset.test_mask,
                               shuffle=True)
  non_priv_train_loader = NeighborLoader(dataset,
                                         num_neighbors=[config.max_degree] * experiment_vars["r_hop"],
                                         batch_size=config.test_batch_size, 
                                         input_nodes=dataset.train_mask,
                                         shuffle=True)

  # search for alpha
  alpha, gamma = search_for_alpha(dataset.train_mask.sum().item(), sigma, clipping_threshold, experiment_vars)
  row["alpha"] = alpha
  row["gamma"] = gamma

  # setup wandb
  if config.wandb_project is not None:
    import wandb
    wandb.init(project=config.wandb_project, config=row, name="Experiment " + str(row["id"]))

  # setup model
  model = GNN(experiment_vars["encoder_dimensions"], experiment_vars["decoder_dimensions"], 
              experiment_vars["activation"], experiment_vars["r_hop"], experiment_vars["dropout"]).to(config.device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = get_optimizer(experiment_vars, clipping_threshold, sigma, model)

  # train/test
  t = 1
  while True:
    sampled_dataset = sample_edgelists(dataset, experiment_vars["degree_bound"], config.device)
    train_loader = NeighborLoader(sampled_dataset, 
                                  num_neighbors=[-1] * experiment_vars["r_hop"],
                                  batch_size=experiment_vars["batch_size"],
                                  input_nodes=sampled_dataset.train_mask,
                                  shuffle=True)
    curr_epsilon = get_epsilon(gamma, t, alpha, experiment_vars["delta"])
    # train if haven't expended all of budget
    if curr_epsilon < experiment_vars["epsilon"]:
      batch = next(iter(train_loader))
      train(batch, model, loss_fn, optimizer, config.device)
    if t % config.test_stepsize == 0 or curr_epsilon >= experiment_vars["epsilon"]:
      if config.wordy:
        print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:>0.2f} GB, memory allocated: {torch.cuda.memory_allocated(0)/1024**3:>0.2f} GB")
        print("Training step:", t)
      _, train_acc = n_batch_test(non_priv_train_loader, config.train_batches, "TRAIN", model, loss_fn, config.device, wordy=config.wordy)
      _, test_acc = n_batch_test(test_loader, config.train_batches, "TEST", model, loss_fn, config.device, wordy=config.wordy)
      if config.wordy:
        print(" Optimizer Achieves ({:>0.1f}, {})-DP".format(curr_epsilon, experiment_vars["delta"]))
      row["epsilon"] = curr_epsilon
      row["step"] = t
      row["train_acc"] = train_acc
      row["test_acc"] = test_acc
      # write data to csv
      if config.wandb_project is None:
        with open(config.results_path, 'a') as f_object:
          writer_object = writer(f_object)
          writer_object.writerow(row.values())
          f_object.close()
      else:
        wandb.log({"train_acc": row["train_acc"], "test_acc": row["test_acc"], "step": row["step"], "epsilon": row["epsilon"]})

    if curr_epsilon >= experiment_vars["epsilon"]:
      break
    torch.cuda.empty_cache()
    gc.collect()
    # increment iteration
    t = t+1


def estimate(experiment_vars):
  # prepare dataset
  dataset, _ = load_dataset(experiment_vars["dataset"])
  if not hasattr(dataset, 'train_mask'):
    train_test_split(dataset, 0.2)
  # get clipping threshold
  clipping_threshold = get_clipping_threshold(dataset, experiment_vars)
  # get sigma according to the equation in section 6
  sigma = get_sigma(experiment_vars, clipping_threshold)
  # search for alpha
  alpha, gamma = search_for_alpha(dataset.train_mask.sum().item(), sigma, clipping_threshold, experiment_vars)
  # runtime estimate
  if experiment_vars["r_hop"] == 1:
    runtime = 5.9232 * np.power(gamma, -0.9679)
  elif experiment_vars["r_hop"] == 2:
    runtime = 35.0279 * np.power(gamma, -0.9168)
  elif experiment_vars["r_hop"] == 3:
    runtime = 58.6427 * np.power(gamma, -1.0870)
  return alpha, gamma, sigma, runtime


def search_for_alpha(n, sigma, clipping_threshold, experiment_vars):
  # if precomputed
  if "alpha" in experiment_vars and "gamma" in experiment_vars:
    return experiment_vars["alpha"], experiment_vars["gamma"]
  # otherwise
  alpha, gamma = 1.01, np.inf
  for alpha_ in np.linspace(1.01, 60, num=200):
    gamma_ = get_gamma(n, experiment_vars["batch_size"], clipping_threshold, sigma, experiment_vars["r_hop"], 
                      experiment_vars["degree_bound"], alpha_, experiment_vars["delta"])

    if (get_epsilon(gamma_, 1, alpha_, experiment_vars["delta"]) < get_epsilon(gamma, 1, alpha, experiment_vars["delta"])):
      alpha = alpha_
      gamma = gamma_
  return alpha, gamma


# if clipping threshold is set to zero at configuration, then use the percentile (adaptive choice)
def get_clipping_threshold(dataset, experiment_vars):
  if "clipping_threshold" not in experiment_vars:
    model = GNN(experiment_vars["encoder_dimensions"], experiment_vars["decoder_dimensions"], 
                experiment_vars["activation"], experiment_vars["r_hop"], experiment_vars["dropout"]).to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    sampled_dataset = sample_edgelists(dataset, experiment_vars["degree_bound"], config.device)
    dataloader = NeighborLoader(sampled_dataset, 
                                num_neighbors=[-1] * experiment_vars["r_hop"],
                                batch_size=experiment_vars["batch_size"],
                                input_nodes=sampled_dataset.train_mask,
                                shuffle=True)
    clipping_threshold = experiment_vars["clipping_multiplier"] * get_gradient_percentile(model, loss_fn, dataloader, experiment_vars["clipping_percentile"])
    torch.cuda.empty_cache()
    gc.collect()
  else:
    clipping_threshold = experiment_vars["clipping_threshold"]
  return clipping_threshold


def get_sigma(experiment_vars, clipping_threshold):
  # if precomputed
  if "sigma" in experiment_vars:
    return experiment_vars["sigma"]
  # otherwise
  return experiment_vars["noise_multiplier"] * 2 * clipping_threshold * get_N(experiment_vars["degree_bound"], experiment_vars["r_hop"])


def get_optimizer(experiment_vars, clipping_threshold, sigma, model):
  optimizer = None
  if experiment_vars["optimizer"] == "DPSGD":
    optimizer = optim.DPSGD(l2_norm_clip=clipping_threshold, noise_multiplier=sigma, batch_size=experiment_vars["batch_size"],
                            params=model.parameters(), lr=experiment_vars["lr"], weight_decay=experiment_vars["weight_decay"])
  elif experiment_vars["optimizer"] == "DPAdam":
    optimizer = optim.DPAdam(l2_norm_clip=clipping_threshold, noise_multiplier=sigma, batch_size=experiment_vars["batch_size"],
                             params=model.parameters(), lr=experiment_vars["lr"], weight_decay=experiment_vars["weight_decay"],
                             betas=(experiment_vars["beta1"],experiment_vars["beta2"]), eps=experiment_vars["eps"])
  elif experiment_vars["optimizer"] == "DPAdamFixed":
    from dp_nlp.adam_corr import AdamCorr
    optimizer = AdamCorr(dp_l2_norm_clip=clipping_threshold, dp_noise_multiplier=sigma, dp_batch_size=experiment_vars["batch_size"],
                         params=model.parameters(), lr=experiment_vars["lr"], weight_decay=experiment_vars["weight_decay"],
                         betas=(experiment_vars["beta1"],experiment_vars["beta2"]), eps_root=experiment_vars["eps"], eps=0)
  elif experiment_vars["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(params=model.parameters(), lr=experiment_vars["lr"], weight_decay=experiment_vars["weight_decay"])
  return optimizer

# ================================================================================================================================================================================================
# SET UP ARG PARSER

parser = argparse.ArgumentParser()
# loggable variables
header = ["batch_size", "epsilon", "delta", "r_hop", "degree_bound", "clipping_threshold", "clipping_multiplier", "clipping_percentile", "noise_multiplier", "lr", \
          "weight_decay", "beta1", "beta2", "eps", "encoder_dimensions", "decoder_dimensions", "activation", "dropout", "optimizer", "dataset", "setup", "sigma", "alpha", "gamma", "step", "train_acc", \
          "test_acc", "id"]
parser.add_argument("--batch_size", help="size of batch", default=10000, type=int)
parser.add_argument("--epsilon", help="privacy budget (for DP)", default=10, type=float)
parser.add_argument("--delta", help="leakage probability (for DP)", default=5e-8, type=float)
parser.add_argument("--r_hop", help="number of hops", default=1, type=int)
parser.add_argument("--degree_bound", help="degree bound", default=10, type=int)
parser.add_argument("--clipping_threshold", help="clipping threshold set manually", default=0.01, type=float)
parser.add_argument("--clipping_multiplier", help="clipping threshold computed as percentile of gradients multiplied by this multiplier", default=1, type=float)
parser.add_argument("--clipping_percentile", help="clipping threshold computed as percentile of gradients", default=0.75, type=float)
parser.add_argument("--noise_multiplier", help="the amount of noise is a function of the sensitivity mutliplied by this multiplier", default=2, type=float)
parser.add_argument("--lr", help="learning rate", default=1e-4, type=float)
parser.add_argument("--weight_decay", help="regularizer strength", default=1e-5, type=float)
parser.add_argument("--encoder_dimensions", help="list of integers as dimensions of encoder (last dimension must match first dimension of decoder)", default=[128, 256, 256], nargs="+", type=int)
parser.add_argument("--decoder_dimensions", help="list of integers as dimensions of decoder (first dimension must match last dimension of encoder)", default=[256, 256, 349], nargs="+", type=int)
parser.add_argument("--activation", help="the name of the activation to use", default="relu", choices = ["relu", "tanh"], type=str)
parser.add_argument("--dropout", help="dropout rate", default=0.1, type=float)
parser.add_argument("--optimizer", help="optimizer to be used", default="DPAdam", choices=["DPSGD", "DPAdam", "DPAdamFixed", "Adam"], type=str)
parser.add_argument("--dataset", help="dataset to be used", default="ogb_mag", choices=["ogb_mag", "reddit"], type=str)
parser.add_argument("--setup", help="setup to be used", default="ours", choices=["original", "ours", "non-dp"], type=str)
parser.add_argument("--sigma", help="precomputed sigma", default=None, type=float)
parser.add_argument("--alpha", help="precomputed alpha", default=None, type=float)
parser.add_argument("--gamma", help="precomputed gamma", default=None, type=float)
parser.add_argument("--beta1", help="beta1 used by Adam", default=0.9, type=float)
parser.add_argument("--eps", help="eps used by Adam", default=1e-8, type=float)
parser.add_argument("--id", help="id of experiment", default=0, type=int)
# environment variables
parser.add_argument("--device", help="which device to use", default="cuda", choices=["cpu", "cuda"], type=str)
parser.add_argument("--results_path", help="path to results file", default="./data/results.csv", type=str)
parser.add_argument("--wordy", help="log everything", action="store_true")
parser.add_argument("--wandb_project", help="save results to wandb in the specified project", type=str)
parser.add_argument("--max_degree", help="the maximum number of neighbours to include when testing model", default=50, type=int)
parser.add_argument("--test_batch_size", help="the batch size used by the (non-dp) test set", default=1000, type=int)
parser.add_argument("--test_stepsize", help="the number of steps between tests/logs", default=100, type=int)
parser.add_argument("--train_batches", help="the number of batches used to compute training average accuracy", default=50, type=int)


def get_experiment_vars_from_args(args):
  return get_experiment_vars(parser.parse_args(args))


def initialize_results_file(config):
  if not os.path.isfile(config.results_path) and config.wandb_project is None:
    with open(config.results_path, 'w') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(header)
        f_object.close()


def get_experiment_vars(config):
  experiment_vars = {}
  config_dict = vars(config)
  for key in header:
    if key in config_dict and config_dict[key] is not None:
      experiment_vars[key] = config_dict[key]
  experiment_vars["beta2"] = 1 - (1 - experiment_vars["beta1"])**2

  return experiment_vars


if __name__ == '__main__':
  config = parser.parse_args()
  initialize_results_file(config)
  run_experiment(get_experiment_vars(config), config)
