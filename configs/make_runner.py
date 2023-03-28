import os
import argparse
import json


def parse_axis(axis):
    if "combine" in axis:
        return combine_axes(*axis["combine"])
    elif "join" in axis:
        return join_axes(*axis["join"])
    else:
        return axis


def combine_axes(*axes):
    combination = {}
    for axis in axes:
        current_length = 1
        if combination.values():
            current_length = len(list(combination.values())[0])
        axis = parse_axis(axis)
        # combine axis
        added_length = len(list(axis.values())[0])
        for key in combination:
            combination[key] = combination[key]*added_length
        for key in axis:
            combination[key] = [value for value in axis[key] for i in range(current_length)]
    return combination


def join_axes(*axes):
    joined = {}
    for axis in axes:
        current_length = 0
        if joined.values():
            current_length = len(list(joined.values())[0])
        axis = parse_axis(axis)
        # join axis
        added_length = len(list(axis.values())[0])
        for key in joined:
            if key not in axis:
                joined[key] += ["N/a"]*added_length
        for key in axis:
            if key not in joined:
                joined[key] = ["N/a"]*current_length
            joined[key] += axis[key]
    return joined



enums = {
    "setup": ["original", "ours", "non-dp"],
    "optimizer": ["DPSGD", "DPAdam", "DPAdamFixed", "Adam"],
    "dataset": ["ogb_mag", "reddit"]
}
def parse_command(config, id):
    command = "python main.py"
    for key, value in config.items():
        if value != "N/a":
            command += " --" + key + " "
            if key in enums.keys() and isinstance(value, int):
                command += enums[key][value]
            elif key in ["encoder_dimensions", "decoder_dimensions"]:
                command += " ".join(value)
            else:
                command += str(value)
    command += " --id " + str(id)
    return command
    

def parse_environment_variables(json_obj):
    command = ""
    for key, value in json_obj.items():
        if key == "wordy":
            if value == True:
                command += " --" + key 
        elif key != "grid":
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
        grid = parse_axis(json_obj["grid"])
    
    # make runner
    n_configs = len(next(iter(grid.values())))
    with open(args.out_path, 'w') as f_object:
        for i in range(n_configs):
            # get config for this experiment (i)
            config = {}
            for key, value in grid.items():
                config[key] = value[i]
            command = parse_command(config, i+1) + parse_environment_variables(json_obj)
            # write command
            f_object.write(command + "\n")
        f_object.close()
