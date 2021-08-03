import math
import torch
from torch import nn
from functools import partial
from random_fourier_features import RandomFourierFeatures
_SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')
_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian', 'laplacian']



class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class CustomRandomFeatureLayer(nn.Module):
    """
    Allows users to input custom functions to simulate a kernel.
    The idea is to approximate a kernel function K(x_i, x_j) via a decomposition
    K_ij = phi_i @ phi_j, given some nonlinearity phi_i(x). If phi is a probabilistic
    mapping with proper statistics, we can approximate K_ij for things like RBF Gaussians.

    This class allows the user to define the phi function

    Args:
        in_features: input feature dimension, x.shape[-1]
        out_featuers: size of random features phi.shape[-1]
        kernel_init: a torch.nn.init object, defaults to torch.nn.init.normal_
        bias_init: a torch.nn.init object, defaults to torch.nn.init.uniform_
        activation: a torch.nn.init object, defaults to cosine
    Returns:
        phi: a tensor of shape (..., random_feature_dimension)
    """
    def __init__(
        self,
        in_features,
        out_features,
        kernel_init=None,
        bias_init=None,
        activation=None
    ):
        super(CustomRandomFeatureLayer, self).__init__()
        if kernel_init is None:
            kernel_init = nn.init.normal_
        if bias_init is None:
            bias_init = partial(nn.init.uniform_, a=0, b=2 * math.pi)
        if activation is None:
            activation = torch.cos

        self.weight = kernel_init(
            nn.Parameter(
                torch.empty(in_features, out_features),
                requires_grad=False
            )
        )
        self.bias = bias_init(
            nn.Parameter(
                torch.empty(out_features),
                requires_grad=False
            )
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(x @ self.weight + self.bias)


class RandomFeatureGaussianProcess(nn.Module):
    """Gaussian process layer with random feature approximation [1]. Based heavily
    on the TensorFlow implementation [2].

    During training, the model updates the maximum a posteriori (MAP) logits
    estimates and posterior precision matrix using minibatch statistics. During
    inference, the model divides the MAP logit estimates by the predictive
    standard deviation, which is equivalent to approximating the posterior mean
    of the predictive probability via the mean-field approximation.

    User can specify different types of random features by setting
    `use_custom_random_features=True`, and changing the initializer and activations
    of the custom random features. For example:

        MLP Kernel: initializer='random_normal', activation=tf.nn.relu
        RBF Kernel: initializer='random_normal', activation=tf.math.cos

    A linear kernel can also be specified by setting gp_kernel_type='linear' and
    `use_custom_random_features=True`.

    [1]: Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
             Machines. In _Neural Information Processing Systems_, 2007.
             https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    [2]: https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/gaussian_process.py

    Attributes:
        units: (int) The dimensionality of layer.
        num_inducing: (int) The number of random features for the approximation.
        is_training: (tf.bool) Whether the layer is set in training mode. If so the
            layer updates the Gaussian process' variance estimate using statistics
            computed from the incoming minibatches.

        The scale_random_features param scales Phi by 2. / sqrt(num_inducing) following [1].
        When using GP layer as the output layer of a nerual network,
        it is recommended to turn this scaling off to prevent it from changing
        the learning rate to the hidden layers.
    """
    def __init__(
        self,
        in_features,
        out_features,
        random_features=1024,
        normalize_input=False,
        scale_random_features=True,
        return_random_features=False,
        return_covariance=True,
        kernel_type='gaussian',
        kernel_scale=1.0,
        init_output_bias=0.0,
        kernel_scale_trainable=False,
        output_bias_trainable=False,
        use_custom_random_features=False,
        covariance_momentum=0.999,
        covariance_ridge_penalty=1.0,
        covariance_likelihood='gaussian',
        custom_random_features_initializer=None,
        custom_random_features_activation=None,
        return_dict=True,
    ):
        """Initializes a random-feature Gaussian process layer instance.

        Args:
            in_features: (int) Number of input units.

            out_features: (int) Number of output units.

            random_features: (int) Number of random Fourier features used for
                approximating the Gaussian process.

            kernel_type: (string) The type of kernel function to use for Gaussian
                process. Currently defaults to 'gaussian' which is the Gaussian RBF
                kernel.

            kernel_scale: (float) The length-scale parameter of the a
                shift-invariant kernel function, i.e., for RBF kernel:
                exp(-|x1 - x2|**2 / 2 * kernel_scale).

            output_bias: (float) Scalar initial value for the bias vect
            or.

            normalize_input: (bool) Whether to normalize the input to Gaussian
                process.

            kernel_scale_trainable: (bool) Whether the length scale variable is
                trainable.

            output_bias_trainable: (bool) Whether the bias is trainable.

            cov_momentum: (float) A discount factor used to compute the moving
                average for posterior covariance matrix.

            cov_ridge_penalty: (float) Initial Ridge penalty to posterior
                covariance matrix.

            scale_random_features: (bool) Whether to scale the random feature
                by sqrt(2. / num_inducing).

            use_custom_random_features: (bool) Whether to use custom random
                features implemented using tf.keras.layers.Dense.

            custom_random_features_initializer: (callable) Initializer for
                the random features. Default to random normal which approximates a RBF
                kernel function if activation function is cos.

            custom_random_features_activation: (callable) Activation function for the
                random feature layer. Default to cosine which approximates a RBF
                kernel function.

            l2_regularization: (float) The strength of l2 regularization on the output
                weights.

            covariance_likelihood: (string) Likelihood to use for computing Laplace
                approximation for covariance matrix. Default to `gaussian`.

            return_covariance: (bool) Whether to also return GP covariance matrix.
                If False then no covariance learning is performed.

            return_random_features: (bool) Whether to also return random features.
        """
        super(RandomFeatureGaussianProcess, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.random_features = random_features

        self.normalize_input = normalize_input

        self.input_scale = 1. / math.sqrt(kernel_scale)
        self.feature_scale = math.sqrt(2. / float(random_features))
        self.scale_random_features = scale_random_features

        self.return_random_features = return_random_features
        self.return_covariance = return_covariance

        self.kernel_type = kernel_type
        self.kernel_scale = kernel_scale
        self.init_output_bias = init_output_bias
        self.kernel_scale_trainable = kernel_scale_trainable
        self.output_bias_trainable = output_bias_trainable

        self.use_custom_random_features = use_custom_random_features
        self.custom_random_features_initializer = custom_random_features_initializer
        self.custom_random_features_activation = custom_random_features_activation

        self.covariance_momentum = covariance_momentum
        self.covariance_ridge_penalty = covariance_ridge_penalty
        self.covariance_likelihood = covariance_likelihood

        self.return_dict = return_dict

        if normalize_input:
            self.input_layernorm = nn.LayerNorm(in_features)

        if not use_custom_random_features:
            self.random_feature_layer = RandomFourierFeatures(
                in_features=self.in_features,
                out_features=self.random_features,
                kernel_type=self.kernel_type,
                kernel_scale=self.kernel_scale,
                kernel_scale_trainable=self.kernel_scale_trainable,
            )

        # Make random feature layer
        elif kernel_type.lower() == 'linear':
            self.random_features = random_features = in_features
            self.feature_scale = math.sqrt(2. / float(self.random_features))
            self.random_feature_layer = IdentityLayer()

        else:
            self.random_feature_layer = CustomRandomFeatureLayer(
             in_features=self.in_features,
             out_features=self.random_features,
             kernel_init=self.custom_random_features_initializer,
             bias_init=None,
             activation=self.custom_random_features_activation
        )

        if return_covariance:
            self.covariance_layer = LaplaceRandomFeatureCovariance(
                in_features=self.random_features,
                momentum=covariance_momentum,
                ridge_penalty=covariance_ridge_penalty,
                likelihood=covariance_likelihood,
            )

        self.output_layer = nn.Linear(
            self.random_features,
            self.out_features,
            bias=False
        )

        self.output_bias = nn.Parameter(
            torch.ones(out_features) * init_output_bias,
            requires_grad=output_bias_trainable,
        )

    def reset_precision(self):
        self.covariance_layer.reset_precision()

    def forward(self, x):
        if self.normalize_input:
            x = self.input_layernorm(x)

        elif self.use_custom_random_features:
            x = x * self.input_scale

        Phi = self.random_feature_layer(x)

        if self.scale_random_features:
            Phi = Phi * self.feature_scale

        logits = self.output_layer(Phi) + self.output_bias

        if self.return_covariance:
            covariance = self.covariance_layer(Phi, logits).to(Phi.device)

        if not self.return_dict:
            res = (logits, )
            if self.return_covariance:
                res += (covariance,)
            if self.return_random_features:
                res += (Phi,)
            return res

        model_output = {'logits':logits}

        if self.return_covariance:
            model_output['covariance'] = covariance

        if self.return_random_features:
            model_output['random_features'] = Phi

        return model_output


class LaplaceRandomFeatureCovariance(nn.Module):
    """Computes the Gaussian Process covariance using Laplace method.

    At training time, this layer updates the Gaussian process posterior using
    model features in minibatches.

    Attributes:
        momentum: (float) A discount factor used to compute the moving average for
            posterior precision matrix. Analogous to the momentum factor in batch
            normalization. If -1 then update covariance matrix using a naive sum
            without momentum, which is desirable if the goal is to compute the exact
            covariance matrix by passing through data once (say in the final epoch).

        ridge_penalty: (float) Initial Ridge penalty to weight covariance matrix.
            This value is used to stablize the eigenvalues of weight covariance
            estimate so that the matrix inverse can be computed for Cov = inv(t(X) * X
            + s * I). The ridge factor s cannot be too large since otherwise it will
            dominate the t(X) * X term and make covariance estimate not meaningful.

        likelihood: (str) The likelihood to use for computing Laplace approximation
            for the covariance matrix. Can be one of ('binary_logistic', 'poisson',
            'gaussian').
    """

    def __init__(
        self,
        in_features,
        momentum=0.999,
        ridge_penalty=1.,
        likelihood='gaussian',
        device=None,
        dtype=None,
    ):
        if likelihood not in _SUPPORTED_LIKELIHOOD:
            raise ValueError(
                f'"likelihood" must be one of {_SUPPORTED_LIKELIHOOD}, got {likelihood}.'
            )

        super(LaplaceRandomFeatureCovariance, self).__init__()
        self.in_features = in_features
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.likelihood = likelihood

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.dtype = dtype

        self.init_precision = ridge_penalty * torch.eye(in_features, **self.factory_kwargs)

        self.register_buffer('_precision', ridge_penalty * torch.eye(in_features, **self.factory_kwargs))
        self.register_buffer('_covariance', ridge_penalty * torch.eye(in_features, **self.factory_kwargs))
        self.covariance_is_cached = False

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, val):
        self.covariance_is_cached = False
        # self._precision = self.register_buffer('_precision', val)
        self._precision = val

    @property
    def covariance(self):
        if not self.covariance_is_cached:
            self._covariance = torch.linalg.inv(self._precision)
            self.covariance_is_cached = True
        return self._covariance

    @covariance.setter
    def covariance(self, val):
        # self._covariance = self.register_buffer('_covariance', val)
        self._covariance = val

    def update_precision(self, Phi, logits):
        """
        Given the current forward pass yielding random features Phi, update the covariance matrix
        for the entire set of input data.
        """
        if self.likelihood != 'gaussian':
            if logits is None:
                raise ValueError(f'"logits" cannot be None when likelihood={self.likelihood}')
            if logits.shape[-1] != 1:
                raise ValueError(
                        f'likelihood={self.likelihood} only supports univariate logits.'
                        f'Got logits dimension: {logits.shape[-1]}')

        batch_size = Phi.shape[0]

        if self.likelihood == 'binary_logistic':
            prob = torch.sigmoid(logits)
            prob_multiplier = prob * (1. - prob)
        elif self.likelihood == 'poisson':
            prob_multiplier = torch.exp(logits)
        elif self.likelihood == 'gaussian':
            prob_multiplier = torch.ones(1, device=Phi.device) * 1.
        else:
            raise ValueError(f"Invalid likelihood entered: {self.likelihood}")

        Phi = torch.sqrt(prob_multiplier) * Phi
        batch_precision = torch.transpose(Phi, -2, -1) @ Phi

        # Update the non-batch (i.e. all data) precision matrix.
        if self.momentum > 0:
            batch_precision = batch_precision / batch_size
            self.precision = (
                    self.momentum * self.precision
                    + (1-self.momentum) * batch_precision.to(self.device)
                )
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            self.precision += batch_precision.to(self.device)
        return self

    def reset_precision(self):
        self.precision = self.init_precision.clone()

    def compute_predictive_covariance(self, Phi):
        """Computes posterior predictive variance.

        The given testing random features Phi (B, H), we pull out the covariance matrix
        Cov (H, H) in random feature space and compute var_k = Phi @ Cov @ Phi.T to
        get an (B, B) covariance matrix of the batch with size B.

        Approximates the Gaussian process posterior using random features. Suppose
        the dataset size is N, i.e. there are N datapoints in total. Then the covariance
        would be Cov_train = (Phi_train^T @ Phi_train + lambda * 1)^{-1} = size(H, H), where lambda is
        the ridge regression penalty, and 1 is an (H, H) identity (this is self.init_cov).

        Given a testing batch of size (B', H) of random features, we wish to compute the
        covariance of this test batch given the covariance of the training data. This is done
        via:

        Cov_test = lambda * Phi_test @ Cov_train @ Phi_test^T = size(B', B')

        matrix inversion is expensive, so we cache the precision inverse result. After
        computing any forward pass with training enabled, we reset the covariance matrix
        assuming that the forward pass has made the covariance cache stale (new data came in).

        Args:
            Phi: (torch.tensor) The random feature of testing data to be used for
                computing the covariance matrix. Shape (batch_size, gp_hidden_size).

        Returns:
            (torch.tensor) Predictive covariance matrix, shape (batch_size, batch_size).
        """
        return self.ridge_penalty * Phi @ self.covariance.to(Phi.device) @ torch.transpose(Phi, -2, -1)


    def forward(self, Phi, logits=None):
        """Minibatch updates the GP's posterior precision matrix estimate.

        Args:
            inputs: (tf.Tensor) GP random features, shape (batch_size,
                gp_hidden_size).
            logits: (tf.Tensor) Pre-activation output from the model. Needed
                for Laplace approximation under a non-Gaussian likelihood.
            training: (tf.bool) whether or not the layer is in training mode. If in
                training mode, the gp_weight covariance is updated using gp_feature.

        Returns:
            gp_stddev (tf.Tensor): GP posterior predictive variance,
                shape (batch_size, batch_size).
        """
        batch_size = Phi.shape[0]
        if self.training:
            self.update_precision(Phi=Phi, logits=logits)
            return torch.eye(batch_size, device=Phi.device)
        return self.compute_predictive_covariance(Phi=Phi)


def mean_field_logits(logits, covariance_matrix=None, mean_field_factor=1.):
    """Adjust the model logits so its softmax approximates the posterior mean [1].

    [1]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
             Jackknife.    _arXiv preprint arXiv:2006.07584_, 2020.
             https://arxiv.org/abs/2006.07584

    Arguments:
        logits: A float tensor of shape (batch_size, num_classes).
        covariance_matrix: The covariance matrix of shape (batch_size, batch_size).
            If None then it assumes the covariance_matrix is an identity matrix.
        mean_field_factor: The scale factor for mean-field approximation, used to
            adjust the influence of posterior variance in posterior mean
            approximation. If covariance_matrix=None then it is used as the
            temperature parameter for temperature scaling.

    Returns:
        Tensor of adjusted logits, shape (batch_size, num_classes).
    """
    if mean_field_factor is None or mean_field_factor < 0:
        return logits

    # Compute standard deviation.
    if covariance_matrix is None:
        variances = torch.ones(1, device=logits.device) * 1.
    else:
        variances = torch.diag(covariance_matrix)

    # Compute scaling coefficient for mean-field approximation.
    logits_scale = torch.sqrt(1. + variances * mean_field_factor)

    if len(logits.shape) > 1:
        # Cast logits_scale to compatible dimension.
        logits_scale = torch.unsqueeze(logits_scale, axis=-1)

    return logits / logits_scale
