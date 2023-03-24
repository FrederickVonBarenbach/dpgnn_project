# How to use make_runner.py

<br />

## JSON file

<br />

To make a experiment runner we need to make a JSON that we will feed to ```make_runner.py``` to generate the runner.

The JSON will have the following form:

```json
{
  "combine": [
    {"degree_bound": [10, 5], "r_hop": [1, 1]},
    {"setup":  [0, 1], "lr":  [1e-3, 1e-4]}
  ],
  "test_stepsize": 10
}
```

The ```"combine"``` field indicates that we want to get all combinations of the two specified configurations. So, the result will be:

```json
{"degree_bound": [10, 5, 10, 5], "r_hop": [1, 1, 1, 1], "setup":  [0, 0, 1, 1], "lr":  [1e-3, 1e-3, 1e-4, 1e-4]}
```

This is a total of 4 experiments. Each experiment takes a different ```"degree_bound"```, ```"r_hop"```, ```"setup"```, and ```"lr"```. The first experiment will take the first element of each list and set that to the corresponding hyperparameter. 

For hyperparameters that accept strings such as ```"setup"``` or ```"dataset"```, you can use a number corresponding to the index of the possible value (e.g. the ```"original"``` setup would be ```0```) that the hyperparameter can take which can be found at the bottom of this documentation (or you can just use the string).

The ```"test_stepsize"``` field will make it so that the model outputs test results very 10 steps.

<br />

## JSON fields

<br />

Valid JSON fields include:
 - ```"combine"``` accepts a list of configurations (JSON objects) from a which a grid of all possible combinations will be composed
 - ```"device"``` accepts either the string ```"cuda"``` or ```"cpu"```
 - ```"results_path"``` accepts a string which will be where results from experiments will be stored
 - ```"wordy"``` accepts ```true``` (or ```false```) but by default, it will be ```false```
 - ```"compute_canada"``` accepts ```true``` (or ```false```) but by default, it will be ```false```
 - ```"max_degree"``` is the maximum number of neighbours to include when testing the model (int)
 - ```"test_stepsize"``` is the number of steps between tests/logs (int)
 - ```"train_batches"``` is the number of batches to use when computing the training average accuracy (int)
 -   ```"test_batch_size"``` is the batch size used by the (non-dp) test set


<br />

## Generating bash script

<br />

To generate the runner bash script, feed the JSON to ```make_runner.py```, run the command:

```$ python make_runner.py <path-to-json> --out_path "./run.sh" --results_path "./data/results.csv"```

Please note that the ```out_path``` filepath is with respect to ```make_runner.py``` and the ```results_path``` filepath is with respect to ```main.py```. Now you can run the bash script ```run.sh``` to do your experiments!

<br />
<br />
<br />

# Possible value for string hyperparameters

| Hyperparameter  | 0              | 1            | 2                 | 3          |
|-----------------|----------------|--------------|-------------------|------------|
| ```setup```     | ```original``` | ```ours```   | ```non-dp```      |            |
| ```optimizer``` | ```DPSGD```    | ```DPAdam``` | ```DPAdamFixed``` | ```Adam``` |
| ```dataset```   | ```ogb_mag```  | ```reddit``` |                   |            |
