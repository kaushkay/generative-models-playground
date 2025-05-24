import hydra
from omegaconf import DictConfig
from src.train import train_vae_epoch, evaluate_vae
from src.model import VariationalAutoencoder
from src.dataset import get_dataloader
import torch
import wandb

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project, config=dict(cfg)) # type: ignore
    device = torch.device(cfg.experiment.device)

    model = VariationalAutoencoder(
        input_dim=784,
        latent_dim=cfg.experiment.latent_dim,
        encoder_hidden_dims=cfg.experiment.encoder_hidden_dims,
        decoder_hidden_dims=cfg.experiment.decoder_hidden_dims,
        beta=cfg.experiment.beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.learning_rate)

    train_loader = get_dataloader(cfg.experiment.batch_size, train=True)
    val_loader = get_dataloader(cfg.experiment.batch_size, train=False)

    for epoch in range(1, cfg.experiment.epochs + 1):
        train_loss = train_vae_epoch(model, train_loader, optimizer, device, cfg.experiment.beta)
        val_loss = evaluate_vae(model, val_loader, device, cfg.experiment.beta)

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        print(f"[Epoch {epoch}] Train: {train_loss:.4f}  Val: {val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
