import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm
from config import config
from attacks.pgd import pgd_attack
from utils.eval import evaluate


def get_train_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)


def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)


def train(model):
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    best_acc = 0
    patience = 3
    counter = 0
    os.makedirs(config.get("save_dir", "./"), exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            x_adv = pgd_attack(model, inputs, labels, config["epsilon"], config["alpha"], config["pgd_steps"])

            optimizer.zero_grad()
            outputs = model(x_adv)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}: Training Loss = {total_loss:.4f}")

        # Evaluate clean and adversarial accuracy
        clean_acc = evaluate(model, test_loader, config, adversarial=False)
        evaluate(model, test_loader, config, adversarial=True)

        # Early stopping check
        if clean_acc > best_acc:
            best_acc = clean_acc
            counter = 0
            print(f"New best clean accuracy: {best_acc:.2f}% — continuing training")

            # Save best model
            model_path = os.path.join(config["save_dir"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")
        else:
            counter += 1
            print(f"No improvement in clean accuracy ({counter}/{patience})")
            if counter >= patience:
                print("Early stopping triggered.")
                break