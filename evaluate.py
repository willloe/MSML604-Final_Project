import torch
import os
from models.resnet import ResNet18_CIFAR10
from training.trainer import get_test_loader
from utils.eval import evaluate
from config import config

def evaluate_best_model():
    # model = ResNet18_CIFAR10().to(config["device"])
    model = ResNet50_CIFAR10().to(config["device"])
    model_path = os.path.join(config["save_dir"], "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model.eval()
    print(f"Loaded best model from {model_path}")

    test_loader = get_test_loader()

    print("Evaluating Clean Accuracy:")
    evaluate(model, test_loader, config, adversarial=False)

    print("Evaluating Adversarial Accuracy (PGD):")
    evaluate(model, test_loader, config, adversarial=True)

if __name__ == "__main__":
    evaluate_best_model()
