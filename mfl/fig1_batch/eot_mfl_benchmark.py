import os
import sys
import time  # 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm
sys.path.append("../EntropicOptimalTransportBenchmark")
from eot_benchmark.benchmark.gaussian_mixture_benchmark import ConditionalPlan, PotentialCategoricalDistribution

# patch the log probability calculation
def patched_calculate_log_probs(self, x):
    # Build up a list of log_probs.
    # self.log_probs is a tensor of shape [n_components],
    # self.potential_gaussians_distributions is a list of MultivariateNormal objects.
    log_probs = []
    for base_log_prob, mvn_dist in zip(self.log_probs, self.potential_gaussians_distributions):
        lp = base_log_prob + mvn_dist.log_prob(x)  # should be shape [batch_size]
        if lp.dim() == 0:
            lp = lp.unsqueeze(0)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=1)

PotentialCategoricalDistribution.calculate_log_probs = patched_calculate_log_probs
print("Patched ConditionalPlan.calculate_log_probs successfully.")

# # --- Set up data directory and download benchmark data ---
# data_dir = os.path.abspath("eot_benchmark_data")
# os.makedirs(data_dir, exist_ok=True)
# os.environ["EOT_BENCHMARK_DATA"] = data_dir

# def download_benchmark_data(data_dir):
#     import gdown, zipfile
#     os.makedirs(data_dir, exist_ok=True)
#     zip_path = os.path.join(data_dir, "gaussian_mixture_benchmark_data.zip")
#     expected_file = os.path.join(data_dir, "potential_probs_dim_2_eps_1.0.torch")
#     if not os.path.exists(expected_file):
#         print("Expected benchmark file not found; downloading benchmark data...")
#         if not os.path.exists(zip_path):
#             url = "https://drive.google.com/uc?id=1HNXbrkozARbz4r8fdFbjvPw8R74n1oiY"
#             gdown.download(url, zip_path, quiet=False)
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)
#         print("Download and extraction complete!")
#     else:
#         print("Benchmark data already present.")

# download_benchmark_data(data_dir)

from eot_benchmark.benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler, 
    get_test_input_samples,
)
np.set_printoptions(suppress=True, precision=4)
torch.manual_seed(0)
np.random.seed(0)

# --- Set benchmark parameters ---
DIM = 2               # dimensionality
EPS = 1               # entropic regularization epsilon
N_SAMPLES = 2500      # number of samples

# --- Initialize samplers ---
input_sampler = get_guassian_mixture_benchmark_sampler(
    input_or_target="input", dim=DIM, eps=EPS, batch_size=N_SAMPLES, device="cpu", download=True
)
target_sampler = get_guassian_mixture_benchmark_sampler(
    input_or_target="target", dim=DIM, eps=EPS, batch_size=N_SAMPLES, device="cpu", download=True
)
# We also obtain the ground-truth target samples (for computing the target Gaussian)
gt_plan_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(
    dim=DIM, eps=EPS, batch_size=N_SAMPLES, device="cpu", download=True
)

# --- Draw samples ---
# Draw source samples X0 and target samples Y0
X0 = np.array(input_sampler.sample(n_samples=N_SAMPLES), dtype=np.float64)   # shape: (N_SAMPLES, DIM)
Y0 = np.array(target_sampler.sample(n_samples=N_SAMPLES), dtype=np.float64)  # shape: (N_SAMPLES, DIM)

# Also, for the ground-truth coupling we get (X_gt, Y_gt) but we will only use Y0 for BW-UVP:
X_gt, Y_gt = gt_plan_sampler.sample(n_samples=N_SAMPLES)  # each: (N_SAMPLES, DIM)
XY_gt = np.hstack((X_gt, Y_gt))   # shape: (N_SAMPLES, 2*DIM)
print("Sample shapes:", X0.shape, Y0.shape, XY_gt.shape)

# --- Fit Gaussian Mixture Models (GMMs) for density estimation ---
gmm_source = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
gmm_source.fit(X0)
gmm_target = GaussianMixture(n_components=5, covariance_type='full', init_params='kmeans', random_state=0)
gmm_target.fit(Y0)

means_src = gmm_source.means_[0]
cov_src = gmm_source.covariances_[0]
prec_src = np.linalg.inv(cov_src)

means_tgt = gmm_target.means_
covs_tgt = gmm_target.covariances_
precisions_tgt = [np.linalg.inv(C) for C in covs_tgt]

def score_source(x):
    """Compute gradient of log-density of source at x."""
    return - (x - means_src) @ prec_src.T

def score_target(y):
    """Compute gradient of log-density of target at y."""
    resp = gmm_target.predict_proba(y)
    grad = np.zeros_like(y, dtype=np.float64)
    for k, prec_k in enumerate(precisions_tgt):
        diff = y - means_tgt[k]
        grad += resp[:, [k]] * (-(diff) @ prec_k.T)
    return grad

# --- Initialize coupling particles ---
# We set X = X0 and Y = Y0 as initial particles.
X = X0.copy()
Y = Y0.copy()

# --- MFL Hyperparameters ---
num_iter = 2500      # number of Langevin iterations
step_size = 1e-2     # step size

# --- For BW-UVP, compute the target Gaussian from Y0 ---
target_mean = np.mean(Y0, axis=0)
cov_target = np.cov(Y0, rowvar=False)
sqrt_cov_target = sqrtm(cov_target)
sqrt_cov_target = np.real(sqrt_cov_target)

def compute_bw_uvp(Y_generated):
    """
    Compute the plug-in BW-UVP between the generated target samples and the target measure.
    
    BW-UVP = 100 * BW^2 / (0.5 * trace(cov_target)),
    where BW^2 = trace(cov_generated + cov_target - 2*sqrtm(sqrt_cov_target @ cov_generated @ sqrt_cov_target)).
    """
    cov_generated = np.cov(Y_generated, rowvar=False)
    inner_term = sqrt_cov_target @ cov_generated @ sqrt_cov_target
    inner_sqrt = sqrtm(inner_term)
    bw2 = np.trace(cov_generated + cov_target - 2 * inner_sqrt)
    bw2 = np.real(bw2)
    bw_uvp = 100 * bw2 / (0.5 * np.trace(cov_target))
    return bw_uvp

# --- BW-UVP computation at iteration 0 ---
bw_uvp_history = []
dist0 = compute_bw_uvp(Y)
bw_uvp_history.append(dist0)
print(f"Iteration 0: BW-UVP distance = {dist0:.4f}")

# --- Langevin simulation (Mean-Field Langevin iterations) with timing ---
start_time = time.time()  # start timing the loop
for t in range(1, num_iter+1):
    vX = score_source(X) - (1.0/EPS) * (X - Y)
    vY = score_target(Y) + (1.0/EPS) * (X - Y)
    noise_X = np.random.randn(*X.shape)
    noise_Y = np.random.randn(*Y.shape)
    X += step_size * vX + np.sqrt(2 * step_size) * noise_X
    Y += step_size * vY + np.sqrt(2 * step_size) * noise_Y
    if t % 50 == 0 or t == num_iter:
        dist = compute_bw_uvp(Y)
        bw_uvp_history.append(dist)
        print(f"Iteration {t}: BW-UVP distance = {dist:.4f}")
end_time = time.time()  # end timing

total_time = end_time - start_time
print(f"Total simulation time for {num_iter} iterations: {total_time:.4f} seconds")
print(f"Average time per iteration: {total_time/num_iter:.6f} seconds")

final_distance = bw_uvp_history[-1]
print(f"Final BW-UVP distance: {final_distance:.4f}")

bw_uvp_history = np.array(bw_uvp_history, dtype=np.float64)
np.save('bw_uvp_history.npy', bw_uvp_history)
coupling_samples = np.hstack((X, Y))
np.save('mfl_coupling_samples.npy', coupling_samples)

# --- Plot convergence of BW-UVP ---
iters = np.linspace(0, num_iter, len(bw_uvp_history))
plt.figure(figsize=(6,4))
plt.plot(iters, bw_uvp_history, marker='o')
plt.title('BW-UVP Distance Convergence')
plt.xlabel('Iteration')
plt.ylabel('BW-UVP distance')
plt.ylim(bottom=0.0)
plt.grid(True)
plt.tight_layout()
plt.savefig('bw_uvp_convergence_plot.png')
plt.close()
