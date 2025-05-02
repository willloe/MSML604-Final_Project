import torch
import torch.nn.functional as F

def pgd_attack(model, x, y, epsilon, alpha, steps):
    model.eval()
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv.requires_grad = True  # <- DO NOT use requires_grad_(), this ensures fresh grad tracking
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)

        grad = torch.autograd.grad(
            loss, x_adv, only_inputs=True, retain_graph=False, create_graph=False
        )[0]

        # Update adversarial image
        x_adv = x_adv + alpha * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()  # Detach here to avoid graph accumulation

    return x_adv
