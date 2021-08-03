import torch
from torch import nn
from gaussian_process import RandomFeatureGaussianProcess

import inspect

class SNGP(nn.Module):
    """
    A Spectral Noramlized Gaussian Process classification head. Given an input (..., H),
    this applies three layers
    Linear: (..., H) --> (..., H) + activation + dropout
    Projection: (..., H) --> (..., R), for R < H, and R=128 as default
    Random features: (..., R) --> (..., D) for D >> R and D=1024 as default

    In order to approximate the Gaussian Process (GP) kernel, we should have D >> R. This works when
    we reduce the dimension R, which here is done by a linear projection layer.

    See RandomFeatureGaussianProcess documentation for details about the GP kernel settings.

    Args:
        in_features: (int) the last dimension of the input
        num_classes: (int) number of classes to predict on
        reduction_dim: (int) the dimension of the lower dimensional embedding that represents the final hidden state
        classif_dropout: (float) classifier dropout rate. defaults to 0.2
        activation: (str) the name of the pytorch activation function layer (e.g. ReLU, Tanh). Defaults to Tanh.
    Returns:
        A  tuple or dict of logits, covariance, for logits a (..., C) dimensional logits tensor and covariance a
        (batch_size, batch_size) covariance tensor.
    """
    def __init__(
        self,
        in_features,
        num_classes,
        reduction_dim=128,
        classif_dropout=0.2,
        activation='Tanh',
        **kwargs,
    ):
        super(SNGP, self).__init__()

        # these are some good defaults for the gp layer
        gp_kwargs = {
            'random_features': 1024,
            'kernel_scale_trainable': True,
            'covariance_momentum': 0.999,
            'covariance_likelihood': 'gaussian',
            'normalize_input': True,
            'return_dict': False,
            'output_bias_trainable': False,
        }

        for k, v in kwargs.items():
            if k != 'self' and k in inspect.getfullargspec(
                RandomFeatureGaussianProcess.__init__
            ).args:
                gp_kwargs[k] = v

        self.pre_classifier = nn.utils.spectral_norm(nn.Linear(in_features, in_features))
        self.reduce_dim_layer = nn.utils.spectral_norm(nn.Linear(in_features, reduction_dim))
        self.gp_classifier = RandomFeatureGaussianProcess(
            in_features=reduction_dim,
            out_features=num_classes,
            **gp_kwargs,
        )
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.Tanh()
        self.dropout = nn.Dropout(p=classif_dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.pre_classifier(x)))
        x = self.reduce_dim_layer(x)
        return self.gp_classifier(x)
