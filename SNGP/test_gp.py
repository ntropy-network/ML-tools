import torch
import numpy as np
import itertools as it
import gaussian_process
from tqdm import tqdm
import pytest
from gaussian_process import RandomFeatureGaussianProcess, LaplaceRandomFeatureCovariance, mean_field_logits
from random_fourier_features import RandomFourierFeatures


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data= data.astype(np.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def exact_gaussian_kernel(x1, x2):
    """Computes exact Gaussian kernel value(s) for tensors x1 and x2."""
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    x1_squared = torch.sum(x1 ** 2, list(range(1, len(x1.shape))))
    x2_squared = torch.sum(x2 ** 2, list(range(1, len(x2.shape))))
    square = x1_squared[:, None] + x2_squared[None, :] - 2 * x1 @ x2.T
    return torch.exp(-square / 2.)


def _generate_normal_data(num_sample, num_dim, loc):
    """Generates random data sampled from i.i.d. normal distribution."""
    return np.random.normal(
        size=(num_sample, num_dim),
        loc=loc,
        scale=1. / np.sqrt(num_dim)
    ).astype(np.float32)

def _generate_rbf_data(x_data, orthogonal=True):
    """Generates high-dim data that are the eigencomponents of an RBF kernel."""
    k_rbf = exact_gaussian_kernel(x_data, x_data)
    x_orth, x_diag, _ = np.linalg.svd(k_rbf)
    if orthogonal:
        return x_orth
    return np.diag(np.sqrt(x_diag)) @ x_orth.T

def _make_minibatch_iterator(data_numpy, batch_size, num_epoch):
    """Makes a tf.data.Dataset for given batch size and num epoches."""
    return it.chain.from_iterable([
        torch.utils.data.DataLoader(Dataset(data_numpy), batch_size=batch_size)
        for _ in range(num_epoch)
    ])

def _compute_posterior_kernel(x_train, x_test, kernel_func, ridge_penalty):
    """Computes the posterior covariance matrix of a Gaussian process."""
    num_sample = x_train.shape[0]
    k_tt_inv = np.linalg.inv(kernel_func(x_train, x_train) + ridge_penalty * np.eye(num_sample, dtype=np.float32))
    k_ts = kernel_func(x_train, x_test)
    k_ss = kernel_func(x_test, x_test)

    return k_ss - k_ts.T @ k_tt_inv @ k_ts

num_data_dim = 10
num_inducing = 2048
num_train_sample = 1600
num_test_sample = 256
prec_tolerance = {'atol': 1e-3, 'rtol': 5e-2}
cov_tolerance = {'atol': 5e-2, 'rtol': 2.}

rbf_kern_func = exact_gaussian_kernel

x_tr = _generate_normal_data(num_train_sample, num_data_dim, loc=0.)
x_ts = _generate_normal_data(num_test_sample, num_data_dim, loc=1.)

def test_save_load_random_fourier_features(tmp_path):
    name = tmp_path / 'model.bin'
    model = RandomFourierFeatures(1, 10)
    inp = torch.randn(20, 1)
    prev = model(inp)
    torch.save(model.state_dict(), name)
    model = RandomFourierFeatures(1, 10)
    model.load_state_dict(torch.load(name))
    curr = model(inp)
    np.testing.assert_allclose(prev.detach().numpy(), curr.detach().numpy())

# @pytest.mark.skip(reason='takes too long, and it works')
@pytest.mark.parametrize('generate_orthogonal_data', [(False,), (True,)], ids=['rbf_kernel', 'orthogonal'])
def test_laplace_covariance_minibatch(generate_orthogonal_data):
    """Tests if model correctly learns population-level precision matrix."""
    batch_size = 64
    epochs = 1000
    x_data = _generate_rbf_data(x_ts, generate_orthogonal_data)
    data_iterator = _make_minibatch_iterator(x_data, batch_size, epochs)

    # Estimates precision matrix using minibatch.
    cov_estimator = gaussian_process.LaplaceRandomFeatureCovariance(
        in_features=x_data.shape[-1],
        momentum=0.999,
        ridge_penalty=0,
    ).train()

    for minibatch_data in tqdm(data_iterator, total=epochs * num_test_sample // batch_size):
        cov_estimator(minibatch_data)

    # Evaluation
    prec_mat_expected = x_data.T @ x_data
    prec_mat_computed = (cov_estimator.precision.numpy() * num_test_sample)

    np.testing.assert_allclose(prec_mat_computed, prec_mat_expected, **prec_tolerance)

def test_random_feature_prior_approximation():
    """Tests random feature GP's ability to approximate the exact GP prior."""
    random_features = 10240
    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(
        in_features=x_tr.shape[-1],
        out_features=1,
        random_features=random_features,
        normalize_input=False,
        kernel_type='gaussian',
        return_random_features=True,
        use_custom_random_features=False,
    )

    # Extract random features
    rfgp_model.train()
    gp_feature = rfgp_model(torch.tensor(x_tr).float())['random_features']
    rfgp_model.eval()
    gp_feature_np = gp_feature.detach().numpy()

    prior_kernel_computed = gp_feature_np @ gp_feature_np.T
    prior_kernel_expected = rbf_kern_func(x_tr, x_tr)
    np.testing.assert_allclose(prior_kernel_computed, prior_kernel_expected, **cov_tolerance)

def test_random_feature_posterior_approximation():
    """Tests random feature GP's ability in approximating exact GP posterior."""
    # Set momentum = 0.5 so posterior precision matrix is 0.5 * (I + K).
    gp_cov_momentum = 0.5
    gp_cov_ridge_penalty = 1.
    random_features = 1024

    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(
        in_features=x_tr.shape[-1],
        out_features=1,
        random_features=random_features,
        normalize_input=False,
        kernel_type='gaussian',
        covariance_momentum=gp_cov_momentum,
        covariance_ridge_penalty=gp_cov_ridge_penalty)

    # Computes posterior covariance on test data.
    rfgp_model.train()
    rfgp_model(torch.tensor(x_tr))
    rfgp_model.eval()
    gp_cov_ts = rfgp_model(torch.tensor(x_ts))['covariance']

    # Scale up covariance estimate since prec matrix is down-scaled by momentum.
    post_kernel_computed = gp_cov_ts * gp_cov_momentum
    post_kernel_expected = _compute_posterior_kernel(
        x_tr.astype(np.float32), x_ts.astype(np.float32), rbf_kern_func, gp_cov_ridge_penalty
    )
    np.testing.assert_allclose(post_kernel_computed, post_kernel_expected, **cov_tolerance)

def test_random_feature_linear_kernel():
    """Tests if linear kernel indeed leads to an identity mapping."""
    # Specify linear kernel
    gp_kernel_type = 'linear'
    normalize_input = False
    scale_random_features = False
    use_custom_random_features = True

    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(
        in_features=x_tr.shape[-1],
        out_features=1,
        normalize_input=normalize_input,
        kernel_type=gp_kernel_type,
        scale_random_features=scale_random_features,
        use_custom_random_features=use_custom_random_features,
        return_random_features=True)

    gp_feature = rfgp_model.train()(torch.tensor(x_tr))['random_features']

    # Check if linear kernel leads to identity mapping.
    np.testing.assert_allclose(gp_feature, x_tr, **prec_tolerance)

def test_no_matrix_update_during_test():
    """Tests that the precision matrix is not updated during testing."""
    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(x_tr.shape[-1], 1)

    # Training.
    gp_covmat_null = rfgp_model.train()(torch.tensor(x_tr))['covariance']
    precision_mat_before_test = rfgp_model.covariance_layer.precision

    # Testing.
    rfgp_model.eval()(torch.tensor(x_ts))
    precision_mat_after_test = rfgp_model.covariance_layer.precision

    np.testing.assert_allclose(
            gp_covmat_null, torch.eye(num_train_sample), atol=1e-4)
    np.testing.assert_allclose(
            precision_mat_before_test, precision_mat_after_test, atol=1e-4)

def test_save_load_gaussian_process(tmp_path):
    name = tmp_path / 'model.bin'

    model = gaussian_process.RandomFeatureGaussianProcess(x_tr.shape[-1], 1)

    gp_covmat_null = model.train()(torch.tensor(x_tr))['covariance']
    covariance_before = model.eval()(torch.tensor(x_ts))['covariance']

    torch.save(model.state_dict(), name)
    model = gaussian_process.RandomFeatureGaussianProcess(x_tr.shape[-1], 1)
    model.load_state_dict(torch.load(name))
    covariance_after = model.eval()(torch.tensor(x_ts))['covariance']

    np.testing.assert_allclose(covariance_before.detach().numpy(), covariance_after.detach().numpy())


def tes_mean_field_logits_likelihood():
    """Tests if scaling is correct under different likelihood."""
    batch_size = 10
    num_classes = 12
    variance = 1.5
    mean_field_factor = 2.

    rng = np.random.RandomState(0)
    logits = torch.randn(batch_size, num_classes)
    covmat = torch.diag([variance] * batch_size)

    logits_logistic = mean_field_logits(
            logits, covmat, mean_field_factor=mean_field_factor)

    np.testing.assert_allclose(logits_logistic, logits / 2., atol=1e-4)

def test_mean_field_logits_temperature_scaling():
    """Tests using mean_field_logits as temperature scaling method."""
    batch_size = 10
    num_classes = 12

    rng = np.random.RandomState(0)
    logits = torch.tensor(np.random.randn(batch_size, num_classes))

    # Test if there's no change to logits when mean_field_factor < 0.
    logits_no_change = mean_field_logits(
            logits, covariance_matrix=None, mean_field_factor=-1)

    # Test if mean_field_logits functions as a temperature scaling method when
    # mean_field_factor > 0, with temperature = sqrt(1. + mean_field_factor).
    logits_scale_by_two = mean_field_logits(
            logits, covariance_matrix=None, mean_field_factor=3.)

    np.testing.assert_allclose(logits_no_change, logits, atol=1e-4)
    np.testing.assert_allclose(logits_scale_by_two, logits / 2., atol=1e-4)
