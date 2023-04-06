constants = {
    # "device": "cuda",
    # "delta": 0.00000005,
    # "noise_multiplier": 2,
    # "clipping_multiplier": 1,
    # "weight_decay": 1e-5,
    "setup": "original",
    # "encoder_dimensions": "[602, 256]",
    # "decoder_dimensions": "[256, 41]",
    # "dropout": 0.1,
    # "optimizer": "DPAdam"
    "dataset": "ogb_mag"
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

    # "columns": {"lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}

    # "rows": {"lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
    # "columns": {"noise_multiplier": [1.5, 2, 3, 4]},
    # "best": []

    # "rows": {"eps": [1e-9, 1e-8, 5e-8, 1e-7, 5e-7]},

    "columns": {"lr": [0.0003, 0.0005, 0.0007, 0.003]},
    "rows": {"beta1": [0.6, 0.7]},
    "best": ["eps"]

    # "columns": {"optimizer": ["DPAdam", "DPAdamFixed", "DPSGD"]},
    # "best": ["eps"]
}

# filepath = "f-shpilevskiy/gnn_adam_corr"
filepath = "data/f-shpilevskiy_gnn_adam_corr.zip"
storepath = "./figs/line_opt2_lr_beta1"
epsilon_bound = 10
# using_wandb = True
using_wandb = False

# graph_type = "best_heatmap"
# axes = ("lr", "beta1")
# values = ["test_acc"]
# value_range = [0.1, 0.32]

graph_type = "line"
axes = ("step", "acc")
values = ["test_acc", "train_acc"]
comparisons = {"optimizer": ["DPAdam", "DPAdamFixed"]}
value_range = [0, 0.32]

# comparisons = {"optimizer": ["DPAdam", "DPAdamFixed", "DPSGD"]}
# comparisons = make_grid({"setup": [possible_setups[0]]},
#                         {"optimizer": [possible_optimizer[1], possible_optimizer[2]]})