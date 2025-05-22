import torch
import torch.nn.functional as F

def frank_wolfe_attack(model, x, y, epsilon, steps):
    """
    Frank–Wolfe inner-maximization over an l_inf ball.
    """
    model.eval()
    delta = torch.zeros_like(x).to(x.device)
    for t in range(steps):
        delta.requires_grad_(True)
        outputs = model(x + delta)
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, delta)[0]
       
        s = -epsilon * grad.sign()
        gamma = 2.0 / (t + 2)
        delta = ((1 - gamma) * delta + gamma * s).detach()
    return (x + delta).clamp(0, 1)