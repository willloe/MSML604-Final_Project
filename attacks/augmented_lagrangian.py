import torch
import torch.nn.functional as F

def augmented_lagrangian_attack(model, x, y, epsilon, alpha, steps, mu=1.0):
    """
    Augmented Lagrangian inner-max over l_inf ball via penalty + multiplier updates.
    """
    model.eval()
    delta = torch.zeros_like(x, device=x.device)
    lam = torch.zeros(1, device=x.device)

    for _ in range(steps):
        delta.requires_grad_(True)

        outputs = model(x + delta)
        
        loss = F.cross_entropy(outputs, y)

        c = (delta.view(delta.size(0), -1).abs().max(dim=1)[0] - epsilon).clamp(min=0)
        c_mean = c.mean()

        lagrangian = loss + lam * c_mean + (mu / 2) * c_mean**2

        grad = torch.autograd.grad(lagrangian, delta)[0]
        delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon).detach()

        lam = (lam + mu * c_mean).clamp(min=0)

    return (x + delta).clamp(0, 1)