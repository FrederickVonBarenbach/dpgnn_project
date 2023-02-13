possible_setups = ["original", "ours", "non-dp"]

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
    "batch_size": 10000,
    "epsilon": 15,
    "delta": 0.00000005,
    "degree_bound": 10,
    "clipping_threshold": 0.005, # how should I get C?
    "noise_multiplier": 2,
    "lr": 1e-3, 
    "weight_decay": 1e-5,
    "setup": possible_setups[1],
    "encoder_dimensions": [128, 256, 256],
    "decoder_dimensions": [256, 256, 349],
    "dropout": 0.1,
    "dataset": "ogb_mag"
}

base = {
    "degree_bound": [10, 5, 5, 3, 3, 3],
    "r_hop": [1, 1, 2, 1, 2, 3]
}

experiments = make_grid(base,
                        {"lr": [1e-3, 1e-4, 1e-5]}, 
                        {"noise_multiplier": [2, 4, 5, 6]},
                        {"batch_size": [10000, 5000]},
                        {"setup": [possible_setups[0], possible_setups[1]]})

iterations = 2