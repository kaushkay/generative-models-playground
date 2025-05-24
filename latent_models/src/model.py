import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Tuple, Optional, Dict, Any


class VAEEncoder(nn.Module):
    """Encoder network for Variational Autoencoder.
    Args:
        input_dim: Dimention of input data
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        activation: Activation function to use
        dropout_rate: Dropout Probability
    """

    def __init__(self, input_dim:int, hidden_dims: list, latent_dim:int, activation: str='relu', dropout_rate: float=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        #Build Encoder Layer
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output Layer for mean and log variance
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

        self._initialize_weights()


    def _get_activation(self, activation:str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'elu': nn.ELU,
            'swish': nn.SiLU()
        }

        return activations.get(activation, nn.ReLU())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            mu: Mean of latent dimension (batch_size, latent_dim)
            logvar: Log Variance of latent distribution (batch_size, latent_dim)
        """

        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        return mu, logvar

class VAEDecoder(nn.Module):
    """Decoder network for Variational Autoencoder
    Args:
    latent_dim: Dimension of latent space
    hidden_dims: List of hidden layer dimensions ( reverse order of encoder )
    output_dim: Dimension of output data
    output_activation: Activation for output layer
    activation: Activation function for hidden layers
    droupout_rate: Dropout probability
    """

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, output_activation: str='sigmoid', activation: str='relu', droupout_rate: float=0.0):
        super().__init__()

        # Build decoder layers
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(droupout_rate)
            ])

            prev_dim = hidden_dim

        #Output Layer
        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation:
            layers.append(self._get_activation(output_activation))

        self.decoder = nn.Sequential(*layers)
        self._initialize_weights()



    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'tanh': nn.Tanh(),
            'elu': nn.ELU,
            'swish': nn.SiLU()
        }

        return activations.get(activation, nn.ReLU())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, z:torch.Tensor) -> torch.Tensor:
        """Forward Pass through decoder
        Args:
            z: Latent code tensor of shape (batch_size, latent_dim)

        Returns:
            reconstruction: Reconstructured data (batch_size, outpur_dim)
        """

        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    """
    Complete Variational Autoencoder Implementation

    This implementation supports multiple data types and loss fucntions,
    extensive customization options, and detailed logging capabilities.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_hidden_dims: list,
                 decoder_hidden_dims: list,
                 data_type: str='bernoulli',
                 beta: float=1.0,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.data_type = data_type
        self.beta = beta

        output_activation = {
            'bernoulli': 'sigmoid',
            'gaussian': 'linear',
            'categorical': 'linear'
        }.get(data_type, 'sigmoid')


        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            **kwargs)

        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            output_activation=output_activation,
            **kwargs
        )

        # Prior Distribution
        self.register_buffer('prior_mu', torch.zeros(latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(latent_dim))


    def encode(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode Input to latent distribution parameter"""
        return self.encoder(x)

    def reparameterize(self, mu:torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick to sample from latent distribution"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            return mu + eps * std
        else:
            return mu # Use mean during inference


    def decode(self, z:torch.Tensor) -> torch.Tensor:
        """Decode Latent Code to reconstruction"""
        return self.decoder(z)

    def forward(self, x:torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.
        Returns:
            Dictionary containing:
                - recon_x: Reconstructured Input
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Latent Sample
        """

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        return {
            'recon_x': recon_x,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from the model."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruct(self, x:torch.Tensor) -> torch.Tensor:
        """Reconstruct input data."""
        with torch.no_grad():
            outputs = self.forward(x)
        return outputs['recon_x']

    def get_latent_representation(self, x:torch.Tensor) -> torch.Tensor:
        """Get latent representation of input data"""
        with torch.no_grad():
            mu, logvar = self.encode(x)

        return self.reparameterize(mu, logvar)


