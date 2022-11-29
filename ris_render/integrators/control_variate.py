import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def get_multiindex_2d(degree):
    multiindex = []
    for d in range(degree + 1):
        ar = np.arange(d + 1)
        multiindex.extend(list(zip(ar, d - ar)))
        
    return np.array(multiindex)


class Poly:
    def __init__(self, spatial_size: int, dim: int = 2, degree: int = 1):
        self.spatial_size = spatial_size
        self.dim = dim
        if dim != 2:
            raise NotemplementedError
        self.multiindex = get_multiindex_2d(degree)[1:]
        self.degree = degree
        self.ones = np.ones((spatial_size, len(self.multiindex)))
        
    def feature(self, sample):
        assert sample.shape == (self.spatial_size, self.dim), sample.shape
        feature = jnp.expand_dims(sample, 1).repeat(len(self.multiindex), 1)
        feature = feature ** self.multiindex[None, :]
        return feature
        
    def forward(self, sample):
        feature = self.feature(sample)
        return jnp.einsum('ab,ab->a', self.ones, feature.prod(-1))
    
    def __cal__(self, sample):
        return self.forward(sample)
    
    @property
    def grad(self):
        return grad(lambda x: jnp.einsum('ab,ab->a', self.ones, x.prod(-1)).sum())
    
    @property
    def div(self):
        return lambda y: np.stack([
            grad(lambda x: self.grad(x)[..., 0].sum())(y)[..., 0], 
            grad(lambda x: self.grad(x)[..., 1].sum())(y)[..., 1]
        ], -1)

    
class PolyControlVariate:
    def __init__(self, spatial_size: int, dim: int = 2, degree: int = 1):
        self.spatial_size = spatial_size
        self.dim = dim
        self.degree = degree
        self.poly = Poly(spatial_size, dim, degree)
        self.cov = jnp.zeros((spatial_size, len(self.poly.multiindex), len(self.poly.multiindex)))
        self.mean = jnp.zeros((spatial_size, len(self.poly.multiindex), 3))
        self.n = 0
        self.param = np.zeros((spatial_size, len(self.poly.multiindex), 3))
        
#     def update(self, sample, score, value):
#         value = jnp.array(value)
#         score = jnp.array(score)
#         feature = self.poly.feature(sample)
#         grad = 
#         # feature = self.feature(sample, score)
#         # self.mean = (self.n * self.mean - feature[..., None] * value[:, None, :]) / (self.n + 1)
        
#         self.mean = (self.n * self.mean - score[..., None] * value[:, None, :]) / (self.n + 1)
#         self.cov = ((self.n) * self.cov + np.einsum('ab,ac->abc', score, score)) / (self.n + 1)
#         self.n += 1

    def b(self, sample, score):
        score = jnp.array(score)
        sample = jnp.array(sample)
        feature = self.poly.feature(sample)
        dot = (self.poly.grad(feature) * score[:, None, :]).sum(-1)
        div = self.poly.div(feature).sum(-1)
        res = div + dot
        return res

    def set_mean(self, bs, vals):
        self.mean = ((bs - bs.mean(0)[None, ...])[..., None] * (vals - vals.mean(0)[None, ...])[..., None, :]).mean(0)
        
    def set_cov(self, bs):
        c = (bs - bs.mean(0)[None, ...])
        self.cov = np.einsum('abc,abd->abcd', c, c).mean(0)
        
    def set_param(self):
        inv_cov = np.linalg.pinv(self.cov)
        self.param = np.einsum('abc,abd->acd', inv_cov, self.mean)
    
    def forward(self, sample, score):
        b = self.b(sample, score)
        return np.einsum('abc,ab->ac', self.param, b)
    
    def __call__(self, sample, score):
        return self.forward(sample, score)
    
        