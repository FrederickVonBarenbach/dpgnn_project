import os
import sys
import argparse
import json
import math

sys.path.append('..')
from main import estimate, get_experiment_vars_from_args


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


def parse_args(config, id):
    command = ""
    for key, value in config.items():
        if value != "N/a":
            command += " --" + key + " " + parse_command(key, value)
    command += " --id " + str(id)
    return command

enums = {
    "setup": ["original", "ours", "non-dp"],
    "optimizer": ["DPSGD", "DPAdam", "DPAdamFixed", "Adam"],
    "dataset": ["ogb_mag", "reddit"],
    "activation": ["relu", "tanh"]
}
def parse_command(key, value):
    if key in enums.keys() and isinstance(value, int):
        command = enums[key][value]
    elif key in ["encoder_dimensions", "decoder_dimensions"]:
        command = " ".join(str(item) for item in value)
    else:
        command = str(value)
    return command


def parse_environment_variables(json_obj, config, id):
    command = ""
    for key, value in json_obj.items():
        if key == "wordy" or key == "compute_canada":
            if value == True:
                command += " --" + key 
        elif key != "grid" and key != "iter":
            command += " --" + key + " " + str(value)
    # make experiment name
    if "experiment_name" in json_obj:
        command += " --experiment_name "
        for key in json_obj["experiment_name"]:
            command += key + "=" + parse_command(key, config[key]) + "_"
        command = command[:-1]
    else:
        command += " --experiment_name experiment " + str(id)
    return command


def time_as_str(runtime):
    days = math.floor(runtime/86400)
    runtime -= days*86400
    hours = math.floor(runtime/3600)
    runtime -= hours*3600
    mins = math.ceil(runtime/60)
    return f"{str(days):0>2}-{str(hours):0>2}:{str(mins):0>2}:00"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="input JSON (or previous runner script if recompute is set)", type=str)
    parser.add_argument("--out_path", help="path to runner script", default="../runner.sh", type=str)
    parser.add_argument("--cc", default=None, help="estimate runtime and store as CC runner script at given location", type=str)
    parser.add_argument("--recompute",  action="store_true", help="recompute runtime estimation from previous runner script")
    args = parser.parse_args()

    # if we are recomputing previous runner runtimes
    if args.recompute:
        lines = []
        with open(args.in_path) as runner:
            # group sets of 1 day long experiments (86400)
            total_runtime = 0
            start_line_number = 1
            n = len(runner.readlines())
            runner.seek(0)
            for line_number, line in enumerate(runner):
                print(str(len(lines)+1) + "/" + str(n), end="\r")  
                # get args from command line
                os.chdir('../')
                experiment_vars = get_experiment_vars_from_args(line.split()[2:])
                # recompute
                _, _, _, runtime = estimate(experiment_vars)
                total_runtime += runtime
                os.chdir('configs/')
                # make new cc_command
                if total_runtime > 86400:
                    time_str = time_as_str(total_runtime)
                    if start_line_number == line_number+1:
                        cc_command = f"sbatch --array={start_line_number} --time={time_str} --account=def-mlecuyer cc_executor.sh runner.input\n"
                    else:
                        cc_command = f"sbatch --array={start_line_number}-{line_number+1} --time={time_str} --account=def-mlecuyer cc_executor.sh runner.input\n"
                    start_line_number = line_number+2
                    total_runtime = 0
                    # append line
                    lines.append(cc_command)
        if args.cc is not None:
            # write lines to CC script
            with open(args.cc, 'w') as cc_runner:
                for line in lines:
                    cc_runner.write(line)
        else:
            raise Exception("Should specify CC runner script (--cc <string>) in which to store new recomputed runtimes")
    # otherwise do the usual parse JSON then constrcut runner
    else:
        # parse JSON
        with open(args.in_path) as json_file:
            json_obj = json.load(json_file)
            grid = parse_axis(json_obj["grid"])
        
        # make runner
        n_configs = len(next(iter(grid.values())))
        line = 1
        iterations = 1 if "iter" not in json_obj else json_obj["iter"]

        # open files
        if args.cc is not None:
            cc_runner = open(args.cc, 'w')
        command_file = open(args.out_path, 'w')

        # write to files
        for i in range(n_configs):
            print(str(i+1) + "/" + str(n_configs), end="\r")  
            # get config for this experiment (i)
            config = {}
            for key, value in grid.items():
                config[key] = value[i]
            # precompute runtime if CC
            if args.cc is not None:
                main_args = (parse_args(config, i+1) + parse_environment_variables(json_obj)).split()
                os.chdir('../')
                experiment_vars = get_experiment_vars_from_args(main_args)
                alpha, gamma, sigma, runtime = estimate(experiment_vars)
                os.chdir('configs/')
                config["alpha"] = alpha
                config["gamma"] = gamma
                config["sigma"] = sigma
                time_str = time_as_str(runtime)
                # write line to CC script
                for it in range(iterations):
                    cc_runner.write(f"sbatch --time={time_str} --account=def-mlecuyer cc_executor.sh {line} runner.input\n")
                    line += 1
            # parse command for runner
            command = "python main.py" + parse_args(config, i+1) + parse_environment_variables(json_obj, config, i+1)
            # write command
            command_file.write(iterations * (command + "\n"))

        # close files
        if args.cc is not None:
            cc_runner.close()
        command_file.close()
