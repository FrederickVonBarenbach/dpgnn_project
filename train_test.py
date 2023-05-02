import torch
import gc
import os
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# train non privately
def train_non_priv(batch, model, loss_fn, optimizer, device):
  model.train()
  batch = batch.to(device)
  # compute prediction error
  pred = model(batch)[:batch.batch_size]
  y = batch.y[:batch.batch_size]
  loss = loss_fn(pred, y)
  # backpropagation
  optimizer.zero_grad()
  loss.backward()
  # get clean grad
  clean_grad = get_clean_grad(model)
  # step optimizer
  optimizer.step()
  return clean_grad

# train
def train(train_loader, batch_size, model, loss_fn, optimizer, config, experiment_vars):
  model.train()
  loader_iterator = iter(train_loader)
  # backpropagation
  optimizer.zero_grad()
  for i in range(batch_size):
    batch = next(loader_iterator).to(config.device)
    optimizer.zero_microbatch_grad()
    # compute prediction error
    pred = model(batch)[:batch.batch_size]
    y = batch.y[:batch.batch_size]
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.microbatch_step()
  # get clean grad
  clean_grad = get_clean_grad(model)
  # step optimizer (give histograms if using ADAM based optimizer)
  if experiment_vars["optimizer"] == "DPAdamFixed" or experiment_vars["optimizer"] == "DPAdam":
    _, logging_stats, hist_dict, dummy_step = optimizer.step()
    if hist_dict:
      if not os.path.exists('./data/hist/' + config.experiment_name):
        os.makedirs('./data/hist/' + config.experiment_name)
      pickle.dump(hist_dict, open(('./data/hist/' + config.experiment_name + 'hist_step_{}.pkl'.format(dummy_step)), 'wb'))
  else:
    optimizer.step()
  # get clipped and private grads
  clipped_grad, private_grad = get_clipped_and_private_grad(model)
  return clean_grad, clipped_grad, private_grad


# test
def batch_test(batch, split, model, loss_fn, device, wordy=True):
  size = batch.batch_size
  model.eval()
  test_loss, correct = 0, 0
  with torch.inference_mode():
    batch = batch.to(device)
    pred = model(batch)[:batch.batch_size]
    y = batch.y[:batch.batch_size]
    test_loss += loss_fn(pred, y).item()
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  correct /= size
  if wordy:
    print(f"{split.title()} Error: \n Accuracy: {(100*correct):>0.1f}%, Loss: {test_loss:>8f}")
  # empty gpu
  torch.cuda.empty_cache()
  gc.collect()
  return test_loss, correct


# test
def test(loader, split, model, loss_fn, device, wordy=True):
  size = len(loader)
  model.eval()
  test_loss, correct = 0, 0
  for batch in loader:
    batch_loss, batch_correct = batch_test(batch, split, model, loss_fn, device, wordy=False)
    test_loss += batch_loss
    correct += batch_correct
  correct /= size
  test_loss /= size
  if wordy:
    print(f"{split.title()} Error: \n Avg Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f}")
  return test_loss, correct


# test average over n batches
def n_batch_test(loader, n, split, model, loss_fn, device, wordy=True):
  model.eval()
  test_loss, correct = 0, 0
  for i in range(n):
    batch = next(iter(loader))
    batch_loss, batch_correct = batch_test(batch, split, model, loss_fn, device, wordy=False)
    test_loss += batch_loss
    correct += batch_correct
  correct /= n
  test_loss /= n
  if wordy:
    print(f"{split.title()} Error: \n Avg Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f}")
  return test_loss, correct


def get_clipped_and_private_grad(model):
  param_grad_norms = []
  param_clipped_grad_norms = []
  for param in model.parameters():
      if param.grad is not None:
          param_grad_norms.append(torch.linalg.norm(param.grad))
          param_clipped_grad_norms.append(torch.linalg.norm(param.summed_grad))
  private_grad_norm = torch.linalg.norm(torch.stack(param_grad_norms))
  clipped_grad_norm = torch.linalg.norm(torch.stack(param_clipped_grad_norms))
  return private_grad_norm, clipped_grad_norm


def get_clean_grad(model):
  param_grad_norms = []
  for param in model.parameters():
      if param.grad is not None:
          param_grad_norms.append(torch.linalg.norm(param.grad))
  return torch.linalg.norm(torch.stack(param_grad_norms))