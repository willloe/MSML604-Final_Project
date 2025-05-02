# 🛡️ CIFAR-10 Adversarial Training with PGD

This project implements adversarial training on the CIFAR-10 dataset using **Projected Gradient Descent (PGD)** attacks, following the formulation in **Madry et al. (2018)**. The goal is to build a ResNet-18 classifier that is robust against adversarial perturbations constrained within an ℓ∞-norm ball.

---

## 📁 Project Structure

```
cifar_adversarial_project/
│
├── models/           # ResNet-18 architecture adapted for CIFAR-10
├── attacks/          # PGD attack implementation
├── training/         # Adversarial training loop
├── utils/            # Evaluation helpers
├── config.py         # Hyperparameters and global config
├── main.py           # Entry point to launch training
├── requirements.txt  # Required Python packages
└── README.md         # Project documentation
```

---

## 🧠 Method: Minimax Adversarial Training

We solve the robust optimization problem:

\[
\min_\theta \; \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \max_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y) \right]
\]

- **Inner max**: PGD generates adversarial examples within an ε-ball.
- **Outer min**: model is trained to minimize loss on worst-case examples.

---

## ⚙️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
torch
torchvision
tqdm
```

---

## 🚀 How to Run

Run the training loop with:

```bash
python main.py
```

You can also clone and run this project on **Kaggle Notebooks** with GPU:

```python
!git clone https://github.com/YOUR_USERNAME/cifar-adversarial-project.git
%cd cifar-adversarial-project
!python main.py
```

---

## 📊 Features

- CIFAR-10 dataset (automatically downloaded)
- ResNet-18 backbone
- ℓ∞-norm PGD adversarial attacks
- Modular structure for easy extensions

---

## 🛠️ Future Work

- Evaluate robustness using AutoAttack
- Visualize training and adversarial examples
- Add support for ℓ2-norm attacks
- Save and resume model checkpoints

---

## 📚 Reference

Madry et al., [“Towards Deep Learning Models Resistant to Adversarial Attacks”](https://arxiv.org/abs/1706.06083), ICLR 2018