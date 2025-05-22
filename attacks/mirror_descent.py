import torch
import torch.nn.functional as F

def mirror_descent_attack(model, x, y, epsilon, alpha, steps):
    """
    Mirror Descent inner-maximization over an l_inf ball.
    """
    model.eval()
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon).to(x.device)
    for _ in range(steps):
        delta.requires_grad_(True)
        outputs = model(x + delta)
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, delta)[0]
        # Mirror descent: simple sign-step + projection for l_inf
        delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon).detach()
    return (x + delta).clamp(0, 1)