# How to use make_runner.py

<br />

## JSON file

<br />

To make a experiment runner we need to make a JSON that we will feed to ```make_runner.py``` to generate the runner.

The JSON will have the following form:

```json
{
  "grid": {"combine": [
                {"noise_multiplier": [2]},
                {"join": [
                    {"optimizer": [2],
                     "dataset": [1]},
                    {"combine": [
                        {"eps": [1]},
                        {"lr": [0.3, 0.4]}
                    ]}
                ]}]
          },
  "test_stepsize": 10
}
```

The ```"grid"``` field indicates that we want to create a grid from the following settings. We use two operations: ```"combine"``` and ```"join"```. The ```"combine"``` operation will take all combinations of the specified configurations. The ```"join"``` operation will join the configurations. If some hyperparameter is not present in a configuration that is being joined, we just use the default value indicated by ```"N/a"``` . So, the result will be:

```json
{"noise_multiplier": [2, 2, 2], "optimizer": [2, "N/a", "N/a"], "dataset": [1, "N/a", "N/a"], "eps": ["N/a", 1, 1], "lr": ["N/a", 0.3, 0.4]}
```

This is a total of 4 experiments. Each experiment takes a different ```"degree_bound"```, ```"r_hop"```, ```"setup"```, and ```"lr"```. The first experiment will take the first element of each list and set that to the corresponding hyperparameter. 

For hyperparameters that accept strings such as ```"setup"``` or ```"dataset"```, you can use a number corresponding to the index of the possible value (e.g. the ```"original"``` setup would be ```0```) that the hyperparameter can take which can be found at the bottom of this documentation (or you can just use the string).

The ```"test_stepsize"``` field will make it so that the model outputs test results very 10 steps.

<br />

## JSON fields

<br />

Valid JSON fields include:
 - ```"grid"``` accepts a series of operations (JSON object) which are used to construct the grid used for experiments
 - ```"combine"``` accepts a list of configurations/operations (JSON objects) from a which a combined configuration of all possible combinations will be composed
 - ```"join"``` accepts a list of configurations/operations (JSON objects) from a which a joined configuration will be composed
 - ```"device"``` accepts either the string ```"cuda"``` or ```"cpu"```
 - ```"results_path"``` accepts a string which will be where results from experiments will be stored
 - ```"wordy"``` accepts ```true``` (or ```false```) but by default, it will be ```false```
 - ```"wandb_project"``` accepts a string which is the wandb project in which to save results
 - ```"max_degree"``` is the maximum number of neighbours to include when testing the model (int)
 - ```"test_stepsize"``` is the number of steps between tests/logs (int)
 - ```"train_batches"``` is the number of batches to use when computing the training average accuracy (int)
 - ```"test_batch_size"``` is the batch size used by the (non-dp) test set
 - ```"iter"``` is the number of iterations to do each experiment


<br />

## Generating bash script

<br />

To generate the runner bash script, feed the JSON to ```make_runner.py```, run the command:

```$ python make_runner.py <path-to-json> --out_path "./run.sh" --cc "./cc_runner.sh"```

Please note that the ```out_path``` filepath is with respect to ```make_runner.py``` and the ```results_path``` filepath is with respect to ```main.py```. Now you can run the bash script ```run.sh``` to do your experiments! If you use the ```--cc``` flag, you can generate a set of commands that you can use to run CC experiments. If you use the ```--recompute``` flag, you can instead specify the path to some runner (instead of the json) and then recompute the runtimes from that file into a new CC runner script.

<br />
<br />
<br />

# Possible value for string hyperparameters

| Hyperparameter  | 0              | 1            | 2                 | 3          |
|-----------------|----------------|--------------|-------------------|------------|
| ```setup```     | ```original``` | ```ours```   | ```non-dp```      |            |
| ```optimizer``` | ```DPSGD```    | ```DPAdam``` | ```DPAdamFixed``` | ```Adam``` |
| ```dataset```   | ```ogb_mag```  | ```reddit``` |                   |            |
| ```activation```| ```relu```     | ```tanh```   |                   |            |
