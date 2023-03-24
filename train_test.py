import torch
import gc
from torch_geometric.data import Data

# train
def train(batch, model, loss_fn, optimizer, device):
  model.train()
  batch = batch.to(device)
  # compute prediction error
  pred = model(batch)[:batch.batch_size]
  y = batch.y[:batch.batch_size]
  loss = loss_fn(pred, y)
  # backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


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