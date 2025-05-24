import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Tuple, Optional, Dict, Any
import wandb
import torchvision
import matplotlib.pyplot as plt



def vae_loss_function(recon_x: torch.Tensor,
                      x: torch.Tensor,
                      mu: torch.Tensor,
                      logvar: torch.Tensor,
                      data_type: str = 'bernoulli',
                      beta: float = 1.0,
                      reduction: str='mean') -> Dict[str, torch.Tensor]:
    """Comprehensive VAE loss function supporting multiple data types.

    Args:
        recon_x: Reconstructed Data
        x: Original Data
        mu: Latent mean
        logvar: Latent log variance
        data_type: Type of data ("benaulli', 'gaussian', 'categorical')
        beta: Beta parameter for beta-VAE
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        Dictionary containing individual loss components and total loss.
    """

    batch_size = x.size(0)

    # Reconstruction Loss based on data type
    if data_type == 'bernoulli':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(dim=1) #sum over features
    elif data_type == 'gaussian':
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(dim=1) # sum over features
    elif data_type == 'categorical':
        recon_loss = F.cross_entropy(recon_x, x, reduction='none')
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


    #KL Divergence Loss
    #KL(q(z|x)||p(z)) where p (z) = N(0,I) and q(z|x) = N (mu,sigma^2)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # Apply Reduction
    if reduction == "mean":
        recon_loss = recon_loss.mean()
        kl_loss = kl_loss.mean()

    elif reduction == 'sum':
        recon_loss = recon_loss.sum()
        kl_loss = kl_loss.sum()

    # Total Loss
    total_loss = recon_loss + beta * kl_loss

    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'beta': torch.Tensor(beta)
    }


