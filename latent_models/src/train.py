import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import VariationalAutoencoder
import wandb
from typing import Dict, List, Optional, Any

from .utils import vae_loss_function

def train_vae_epoch(
    model: VariationalAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    data_type: str = "bernoulli",
    beta: float = 1.0,
    max_grad_norm: float=1.0) -> Dict[str, float]: # type: ignore

    """Train VAE for one epoch"""

    model.train()
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss':0.0}
    num_batches = len(dataloader)

    for data, _ in tqdm(dataloader, desc="Training", leave=False):
        data = data.to(device).view(data.size(0), -1)  #Flatten input

        # Forward Pass
        outputs = model(data)

        # Compute Loss
        loss_dict = vae_loss_function(
            recon_x=outputs['recon_x'],
            x=data,
            mu=outputs['mu'],
            logvar=outputs['logvar'],
            data_type=data_type,
            beta=beta,
            reduction='mean'
        )

        # Backword pass and optimization
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Accumulate Losses
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses


def evaluate_vae(
    model: VariationalAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    data_type: str = 'bernoulli',
    beta: float = 1.0
) -> Dict[str, float]:
    """Evaluate VAE on validation dataset."""
    model.eval()
    total_losses = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
    num_batches = len(dataloader)

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Validation", leave=False):
            data = data.to(device).view(data.size(0), -1)  # Flatten input

            outputs = model(data)

            loss_dict = vae_loss_function(
                recon_x=outputs['recon_x'],
                x=data,
                mu=outputs['mu'],
                logvar=outputs['logvar'],
                data_type=data_type,
                beta=beta,
                reduction='mean'
            )

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses

