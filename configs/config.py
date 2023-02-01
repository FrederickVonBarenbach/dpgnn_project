config = {
    "device": "gpu",
    "r_hop": 1,
    "batch_size": 10000,
    "epsilon": 15,
    "delta": 0.00000005,
    "degree_bound": 10,
    "clipping_threshold": 0.005, # how should I get C?
    "noise_multiplier": 2,
    "dataset": "ogb_mag"
}

experiments = {
    "degree_bound": [10, 3, 2],
    "r_hop": [1, 2, 3]
}