import argparse
import os
import torch
from models.cnn import ResNet18_CIFAR10
from training.trainer import train, get_test_loader
from utils.eval import evaluate
from config import config

def load_best_model():
    model = ResNet18_CIFAR10().to(config["device"])
    path = os.path.join(config["save_dir"], "best_model.pth")

    if not os.path.exists(path):
        print(f"No saved model found at: {path}")
        return None

    model.load_state_dict(torch.load(path, map_location=config["device"]))
    print(f"Loaded saved model from {path}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode to run: 'train' or 'eval'")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training mode...")
        model = ResNet18_CIFAR10().to(config["device"])
        train(model)

    elif args.mode == "eval":
        print("Evaluating saved model...")
        model = load_best_model()
        if model:
            test_loader = get_test_loader()
            evaluate(model, test_loader, config, adversarial=False)
            evaluate(model, test_loader, config, adversarial=True)

if __name__ == "__main__":
    main()

# !python main.py --mode train   # for training
# !python main.py --mode eval    # for evaluating saved model