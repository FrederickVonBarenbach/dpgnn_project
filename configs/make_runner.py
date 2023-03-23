import os
import argparse
import json


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
def config_to_command(config, json_obj, id):
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
    command += "--id " + str(id)
    # environment settings
    for key, value in json_obj.items():
        if key == "wordy" or key == "compute_canada":
            if value == True:
                command += " --" + key 
        elif key != "combine":
            command += " --" + key + " " + str(value)
    return command


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
            command = config_to_command(config, json_obj, i+1)
            # write command
            f_object.write(command + "\n")
        f_object.close()
