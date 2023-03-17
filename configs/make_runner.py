import os
import argparse
import json


# TODO: finish the runner.sh generator! Plus make it so that it can make parallel scripts
# logging = True
# test_stepsize = 10

def make_grid(*axes):
    grid = {}
    for axis in axes:
        current_length = 1
        if grid.values():
            current_length = len(list(grid.values())[0])
        added_length = len(list(axis.values())[0])
        for key in grid:
            grid[key] = grid[key]*added_length
        for key in axis:
            grid[key] = [value for value in axis[key] for i in range(current_length)]
    return grid


enums = {
    "setup": ["original", "ours", "non-dp"],
    "optimizer": ["DPSGD", "DPAdam", "DPAdamFixed", "Adam"],
    "dataset": ["ogb_mag", "reddit"]
}
def config_to_command(config, json_obj):
    command = "python main.py"
    # experiment settings
    for key, value in config.items():
        command += " --" + key + " "
        if key in enums.keys() and isinstance(value, int):
            command += enums[key][value]
        elif key in ["encoder_dimensions", "decoder_dimensions"]:
            command += " ".join(value)
        else:
            command += str(value)
    # environment settings
    for key, value in json_obj.items():
        if key == "wordy":
            if value == True:
                command += " --wordy" 
        elif key != "combine":
            command += " --" + key + " " + str(value)
    return command


# config = {
#     "device": "cuda",
#     "batch_size": 1000,
#     "epsilon": 10,
#     "delta": 5e-8,
#     "r_hop": 1,
#     "degree_bound": 10,
#     "clipping_threshold": 0.01,
#     "clipping_multiplier": 1,
#     "clipping_percentile": 0.75,
#     "noise_multiplier": 2,
#     "lr": 1e-4, 
#     "weight_decay": 1e-5,
#     "encoder_dimensions": [128, 256, 256],
#     "decoder_dimensions": [256, 256, 349],
#     "dropout": 0.1,
#     "optimizer": possible_optimizer[1],
#     "dataset": possible_datasets[0],
#     "setup": possible_setups[0],
# }

# if not os.path.isfile(config.results_path):
#     with open(config.results_path, 'w') as f_object:
#         f_object.writelines('echo "I ran this"\n')
#         writer_object = writer(f_object)
#         writer_object.writerow(header)
#         f_object.close()

# experiments = [
#     {"degree_bound": [10, 10], "r_hop": [1, 1], "setup":  [possible_setups[0], possible_setups[1]]}
# ]

# base = {
#     "degree_bound": [10, 5, 3],
#     "r_hop": [1, 2, 3]
# }

# experiments = make_grid(base,
#                         {"lr": [1e-4, 1e-5]}, 
#                         {"noise_multiplier": [3, 4.5]},
#                         {"clipping_percentile": [0.7, 0.8]},
#                         {"batch_size": [10000, 5000]},
#                         {"setup": [possible_setups[0], possible_setups[1], possible_setups[1]],
#                          "optimizer": [possible_optimizer[2], possible_optimizer[1], possible_optimizer[2]]},
#                         {"dataset": [possible_datasets[0]],
#                          "encoder_dimensions": [[128, 256, 256]],
#                          "decoder_dimensions": [[256, 256, 349]]})

# iterations = 1


# base = {
#     "degree_bound": [10, 5, 3],
#     "r_hop": [1, 2, 3]
# }

# experiments = make_grid(base,
#                         {"lr": [1e-4, 1e-5]}, 
#                         {"noise_multiplier": [3, 4.5]},
#                         {"clipping_percentile": [0.7, 0.8]},
#                         {"batch_size": [10000, 5000]},
#                         {"setup": [possible_setups[0], possible_setups[1]]},
#                         {"optimizer": [possible_optimizer[0], possible_optimizer[1], possible_optimizer[2]]},
#                         {"dataset": [possible_datasets[0], possible_datasets[1]],
#                          "encoder_dimensions": [[128, 256, 256], [602, 256, 256]],
#                          "decoder_dimensions": [[256, 256, 349], [256, 256, 41]]})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="input JSON", type=str)
    parser.add_argument("--out_path", help="output bash", default="../runner.sh", type=str)
    args = parser.parse_args()

    # parse JSON
    with open(args.in_path) as json_file:
        json_obj = json.load(json_file)
        grid = make_grid(*json_obj["combine"])
    
    # make runner
    n_configs = len(next(iter(grid.values())))
    with open(args.out_path, 'w') as f_object:
        for i in range(n_configs):
            # get config for this experiment (i)
            config = {}
            for key, value in grid.items():
                config[key] = value[i]
            command = config_to_command(config, json_obj)
            # write command
            f_object.write(command + "\n")
        f_object.close()
