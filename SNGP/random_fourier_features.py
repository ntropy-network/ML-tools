import torch
from torch import nn
import math
_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian', 'laplacian']

class RandomFourierFeatures(nn.Module):
    """
    Given a translation invariant kernel K(x,y) = K(k-y), this layer computes a nonlinear
    function phi: R^m -> R^n such that the kernel may be decomposed as:

    K(x_i, x_j) = phi_i(x_i) @ phi_j(x_j)

    with probability 1 as the dimension of phi -> infinity. Explicity, we compute:

    phi_i = scale * cos(x_j @ W_ji + b_i)

    Args:
        in_features: Positive integer, the dimension of the layer's input, i.e. the
            hidden layer size

        out_features: Positive integer, the dimension of the layer's output, i.e., the
            number of random features used to approximate the kernel.

        kernel_type: (str) The distribution with which to sample W and b. Choose from
            one of 'gaussian' or 'laplacian'. Note that the W and b are not
            trainable.

        kernel_scale: (float>0) The scale factor in the above equation. For Gaussian and Laplacian
            kernels, this represents the correlation length between two points x.
            'gaussian', defaults to `\sqrt(in_features / 2)`, and everything else, defaults
            to 1.0. Both the approximation error of the kernel and the classification quality
            are sensitive to this parameter. If `trainable` is set to `True`, this
            parameter is learned end-to-end during training and the provided value
            serves as the initial value.

            **Note:** When features from this layer are fed to a linear model,
                by making `scale` trainable, the resulting optimization problem is
                no longer convex (even if the loss function used by the linear model
                is convex).

        kernel_scale_trainable: Whether the scaling parameter of the layer should be trainable.
            Defaults to `False`.
    """
    def __init__(
        self,
        in_features,
        out_features,
        kernel_type='gaussian',
        kernel_scale=None,
        kernel_scale_trainable=False,
    ):
        if out_features <= 0:
            raise ValueError(
              '`out_features` should be a positive integer. Given: {}.'.format(out_features)
            )

        if isinstance(kernel_type, str):
            if kernel_type.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
                raise ValueError(
                    'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'
                    .format(kernel_type, _SUPPORTED_RBF_KERNEL_TYPES)
                )

        if kernel_scale is not None and kernel_scale <= 0.0:
            raise ValueError(
                'When provided, `scale` should be a strictly positive float. Given: {}.'.format(kernel_scale)
            )

        super(RandomFourierFeatures, self).__init__()

        if kernel_type.lower() == 'gaussian':
            self.register_buffer('weight', torch.randn(in_features, out_features))
        elif kernel_type.lower() == 'laplacian':
            self.register_buffer('weight', torch.tan(math.pi * (torch.rand(in_features, out_features) - 0.5)))
        else:
            raise ValueError("Select kernel_type from one of ['gaussian', 'laplacian']")

        self.register_buffer('bias', torch.rand(out_features) * 2 * math.pi)

        if kernel_scale is None:
            kernel_scale = math.sqrt(in_features / 2.0) if kernel_type == 'gaussian' else 1.0

        if kernel_scale_trainable:
            self.kernel_scale = nn.Parameter(torch.ones(1) * kernel_scale, requires_grad=True)
        else:
            self.register_buffer('kernel_scale', torch.ones(1) * kernel_scale)

    def forward(self, x):
        kernel_scale = nn.functional.relu(self.kernel_scale)
        weight = self.weight / kernel_scale
        return torch.cos(x @ weight + self.bias)
