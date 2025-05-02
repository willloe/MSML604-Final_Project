import torch
from attacks.pgd import pgd_attack

def evaluate(model, dataloader, config, adversarial=False):
    model.eval()
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

        if adversarial:
            # PGD requires gradients — DO NOT use torch.no_grad() here
            inputs = pgd_attack(
                model,
                inputs,
                labels,
                config["epsilon"],
                config["alpha"],
                config["pgd_steps"]
            )

        # Wrap inference in no_grad to save memory
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    print(f"{'Adversarial' if adversarial else 'Clean'} Accuracy: {accuracy:.2f}%")
    return accuracy