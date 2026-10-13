"""Least-squares AMP discriminator based on WMP's style objective."""

import torch
from torch import nn


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim=60, hidden_dims=(1024, 512)):
        super().__init__()
        layers = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state_pair):
        return self.network(state_pair).squeeze(-1)

    def gradient_penalty(self, expert_pair, coefficient: float):
        with torch.enable_grad():
            inputs = expert_pair.detach().requires_grad_(True)
            logits = self(inputs)
            gradient = torch.autograd.grad(logits.sum(), inputs, create_graph=True)[0]
            return coefficient * gradient.square().sum(dim=-1).mean()
