possible_setups = ["original", "ours", "non-dp"]
possible_optimizer = ["DPSGD", "DPAdam", "DPAdamFixed", "Adam"]
possible_datasets = ["ogb_mag", "reddit"]

logging = True

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

config = {
    "device": "cuda",
    "r_hop": 1,
    "batch_size": 5000,
    "epsilon": 10,
    "delta": 0.00000005,
    "degree_bound": 10,
    "clipping_multiplier": 1,
    "clipping_percentile": 0.75,
    "noise_multiplier": 4,
    "lr": 1e-3, 
    "weight_decay": 1e-5,
    "setup": possible_setups[1],
    "encoder_dimensions": [602, 256, 256],
    "decoder_dimensions": [256, 256, 41],
    "dropout": 0.1,
    "optimizer": possible_optimizer[2],
    "dataset": possible_datasets[1]
}

# experiments = {
#     "degree_bound": [10],
#     "r_hop": [1]
# }

base = {
    "degree_bound": [10, 5, 3],
    "r_hop": [1, 2, 3]
}

experiments = make_grid(base,
                        {"lr": [1e-4, 1e-5]}, 
                        {"noise_multiplier": [3, 4.5]},
                        {"clipping_percentile": [0.7, 0.8]},
                        {"batch_size": [10000, 5000]},
                        {"setup": [possible_setups[0], possible_setups[1]]},
                        {"optimizer": [possible_optimizer[1], possible_optimizer[2]]},
                        {"dataset": [possible_datasets[0], possible_datasets[1]],
                         "encoder_dimensions": [[128, 256, 256], [602, 256, 256]],
                         "decoder_dimensions": [[256, 256, 349], [256, 256, 41]]})

iterations = 1


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
