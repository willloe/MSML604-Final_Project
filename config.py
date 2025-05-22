# config.py

import torch

config = {
  
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-3,

    "epsilon": 8/255,
    "alpha": 2/255,

    "pgd_steps": 7,

    "md_steps": 7,

    "fw_steps": 7,


    "al_steps": 7,
    "al_mu": 1.0,           

    "comp_steps": 7,
    "comp_weights": {       
        "pgd": 1.0,
        "md": 1.0,
        "fw": 1.0,
        "al": 1.0
    },

  
    "attack": "clean",

    "save_dir": "./outputs",

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}