from configs.config import make_grid, possible_setups, possible_datasets, possible_optimizer

constants = {
    # "device": "cuda",
    # "delta": 0.00000005,
    "noise_multiplier": 2,
    # "clipping_multiplier": 1,
    # "weight_decay": 1e-5,
    # "setup": "original",
    # "encoder_dimensions": "[602, 256]",
    # "decoder_dimensions": "[256, 41]",
    # "dropout": 0.1,
    # "optimizer": "DPAdam"
    "dataset": possible_datasets[0]
}

variables = {
    # "rows": {
    #     "degree_bound": [10, 5, 3],
    #     "r_hop": [1, 2, 3]
    # },
    # "columns": make_grid({"lr": [1e-4, 1e-5]}, 
    #                     {"noise_multiplier": [3, 4.5]},
    #                     {"clipping_percentile": [0.7, 0.8]},
    #                     {"batch_size": [10000, 5000]})
    # "rows": {
    #     "r_hop": [1, 2, 2],
    #     "degree_bound": [10, 5, 3]
    # },
    # "columns": {
    #     "setup": ["original", "ours"]
    # },
    # "best": ["lr", "noise_multiplier", "batch_size", "clipping_percentile", "clipping_multiplier"]
}

filepath = "./data/results.csv"
storepath = "./figs/ogb_mag_training_graph"
epsilon_bound = 10

# graph_type = "best_heatmap"
# axes = ("degree_bound", "r_hop")
# values = ["test_acc"]
value_range = [0, 0.3]
graph_type = "line"
axes = ("step", "acc")
values = ["test_acc", "train_acc"]
comparisons = {"setup": [possible_setups[0], possible_setups[1]]}
# comparisons = make_grid({"setup": [possible_setups[0]]},
#                         {"optimizer": [possible_optimizer[1], possible_optimizer[2]]})