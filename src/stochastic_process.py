import numpy as np

class OUProcess(object):
    def __init__(self, a_dim=2, theta=0.15, sigma=0.2, dt=1e-2):
        self.a_dim = a_dim
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self.value = np.zeros(a_dim)
        
    def sample(self):
        dv = - self.theta * self.value * self.dt + self.sigma * np.random.randn(self.a_dim) * np.sqrt(self.dt)
        self.value += dv
        return np.copy(self.value)