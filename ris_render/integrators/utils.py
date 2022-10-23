import numpy as np
import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')


def dr_concat(a, b):
    size_a = dr.width(a)
    size_b = dr.width(b)
    c = dr.empty(type(a), size_a + size_b)
    dr.scatter(c, a, dr.arange(mi.UInt32, size_a))
    dr.scatter(c, b, size_a + dr.arange(mi.UInt32, size_b))
    return c
    
    
class Categorical_np:
    def __init__(self,
                 probs):
        denom = np.sum(probs, axis=-1, keepdims=True)
        self.probs = probs / (denom + 1e-5)
    
    def sample(self):
        return np.argmax(np.apply_along_axis(lambda x: np.random.multinomial(1, pvals=x), axis=-1, arr=self.probs.reshape(-1, self.probs.shape[-1])), 0).reshape(self.probs.shape[:-1])
    
    
class FastCategorical_np:
    def __init__(self, probs: np.ndarray):
        self.probs = probs
    
    def sample(self):
        s = self.probs.cumsum(axis=-1)
        r = np.random.rand(*self.probs.shape[:-1]) * s[..., -1]
        k = (s < r[..., None]).sum(axis=-1)
        return k
    
    
class FastCategorical_dr:
    def __init__(self, weights_cumsum: mi.Float, n_particles: int):
        self.weights_cumsum = weights_cumsum
        self.rng = mi.PCG32(size=dr.width(weights_cumsum) // n_particles)
        self.n_particles = n_particles
        
    def sample(self) -> mi.Float:
        ids = dr.arange(mi.UInt32, self.n_particles - 1, dr.width(self.weights_cumsum), step=self.n_particles)
        weight_sum = dr.gather(mi.Float, self.weights_cumsum, ids)
        r = dr.repeat(self.rng.next_float32() * weight_sum, self.n_particles)
        mask = self.weights_cumsum < r
        ones = dr.ones(mi.Float, dr.width(self.weights_cumsum))
        zeros = dr.zeros(mi.Float, dr.width(self.weights_cumsum)) 
        ids = dr.block_sum(dr.select(mask, ones, zeros), self.n_particles)
        return ids
    