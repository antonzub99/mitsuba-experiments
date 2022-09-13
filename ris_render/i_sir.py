import numpy as np
import tqdm
from scipy.special import logsumexp

from numpy.random import default_rng
from scipy.stats import multivariate_normal

class MulNormal:
    def __init__(self,
                 mean = [0, 0],
                 cov = [[1, 0], [0, 1]],
                 seed = 42
                 ):
        self.mean = mean
        self.cov = cov
        self.rng = default_rng(seed)
    
    def sample(self, size):
        return self.rng.multivariate_normal(self.mean, self.cov, size)
    
    def log_prob(self, x):
        residuals = (x - self.mean)
        return -0.5 * (np.sum(residuals.T * np.dot(np.linalg.inv(self.cov),residuals.T),axis=0) - np.log(np.linalg.det(self.cov)) + 2 * np.log(2 * np.pi))
      
class Categorical_np:
    def __init__(self,
                 logits):
        self.probs = np.exp(logits)
        denom = np.sum(self.probs, axis=-1, keepdims=True)
        self.probs /= denom
    
    def sample(self):
        return np.argmax(np.apply_along_axis(lambda x: np.random.multinomial(1, pvals=x), axis=-1, arr=self.probs), -1)

def log_prob(x):
    log_ps = np.stack([
            np.log(weight) + gauss.log_prob(x) for weight, gauss in zip(weights, gaussians)
            ], axis=0)
    return logsumexp(log_ps, 0)

def i_sir_step(log_target_dens, x_cur, N_part, isir_proposal, return_all_stats=False):
    """
    function to sample with N-particles version of i-SIR
    args:
        N_part - number of particles, integer;
        x0 - current i-sir particle;
    return:
        x_next - selected i-sir particle
    """
    N_samples, lat_size = x_cur.shape
    
    # generate proposals
    proposals = isir_proposal.sample((N_samples, N_part - 1, ))

    # put current particles
    proposals = np.concatenate((x_cur[:, np.newaxis, :], proposals), axis=1)

    # compute importance weights
    log_target_dens_proposals = log_target_dens(proposals.reshape(-1, lat_size)).reshape(N_samples, N_part)
    
    logw = log_target_dens_proposals - isir_proposal.log_prob(proposals.reshape(-1, lat_size)).reshape(N_samples, N_part)
    
    #sample selected particle indexes
    idxs = Categorical_np(logits=logw).sample()

    cur_x = proposals[np.arange(N_samples), idxs]
    
    if return_all_stats:
        return cur_x, proposals, log_target_dens_proposals
    else:
        return cur_x
      
def i_sir(log_target_dens, x0, N_steps, N_part, isir_proposal, seed=42):
    np.random.seed(seed)
    ### sample i-sir

    samples_traj = [x0]
    x_cur = x0
    
    for _ in tqdm.tqdm(range(N_steps)):
        x_cur = i_sir_step(log_target_dens, x_cur, N_part, isir_proposal)
        samples_traj.append(x_cur)

    samples_traj = np.stack(samples_traj).transpose(1, 0, 2)
    
    return samples_traj
