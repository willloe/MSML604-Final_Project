import torch

config = {
    "batch_size": 128,
    "epochs": 10,
    "lr": 1e-3,
    "epsilon": 8/255,
    "alpha": 2/255,
    "pgd_steps": 7,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}