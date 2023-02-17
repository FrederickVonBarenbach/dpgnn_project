from configs.config import make_grid

constants = {
    "device": "cuda",
    "batch_size": 5000,
    "delta": 0.00000005,
    "clipping_multiplier": 1.2,
    "clipping_percentile": 0.75,
    "weight_decay": 1e-5,
    "encoder_dimensions": "[128, 256, 256]",
    "decoder_dimensions": "[256, 256, 349]",
    "dropout": 0.1,
    "optimizer": "DPAdam",
    "dataset": "ogb_mag"
}

variables = {
    "rows": {
        "r_hop": [1, 2, 2],
        "degree_bound": [10, 5, 3]
    },
    "columns": make_grid({"lr": [1e-3, 1e-4]}, {"noise_multiplier": [2, 3, 4]}),
    # "best": ["lr", "noise_multiplier", "batch_size"]
}

filepath = "./data/results.csv"
storepath = "./figs/"
epsilon_bound = 8

# graph_type = "best_heatmap"
# axes = ("degree_bound", "r_hop")
# values = ["test_acc"]
graph_type = "line"
axes = ("step", "acc")
values = ["test_acc", "train_acc"]
comparisons = {"setup": ["ours", "original"]}