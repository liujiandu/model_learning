from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class GPR(object):
    def __init__(self, kername, n_restarts_optimizer, random_state=None):
        if kername=='matern':
            kernel = Matern(nu=2.5)

        self.gpr_ = GaussianProcessRegressor(kernel=kernel,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state = random_state)
    
    def fit(self, x, y):
        return self.gpr_.fit(x,y)

    def predict(self, x, return_std=True):
        return self.gpr_.predict(x, return_std=return_std)

    def set_params(self, **gp_params):
        self.gpr_.set_params(**gp_params)
