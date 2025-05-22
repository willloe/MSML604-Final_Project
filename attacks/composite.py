import torch
import torch.nn.functional as F
from attacks.pgd import pgd_attack
from attacks.mirror_descent import mirror_descent_attack
from attacks.frank_wolfe import frank_wolfe_attack
from attacks.augmented_lagrangian import augmented_lagrangian_attack

def composite_attack(model, x, y, epsilon, alpha, steps, weights=None):
    """
    Composite attack: maximize weighted sum of solver-based losses.
    weights: dict with keys 'pgd','md','fw','al'
    """
    model.eval()
    
    w = weights or {'pgd':1.0, 'md':1.0, 'fw':1.0, 'al':1.0}
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon).to(x.device)
    for _ in range(steps):
        delta.requires_grad_(True)

        loss_pgd = F.cross_entropy(model(x + delta), y)
       
        loss = w['pgd'] * loss_pgd
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon).detach()
    return (x + delta).clamp(0, 1)