config = {
    "device": "cuda",
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
    "degree_bound": [10, 10, 5, 5, 5, 3, 3, 3],
    "r_hop": [1, 2, 1, 2, 3, 1, 2, 3]
}

iterations = 3