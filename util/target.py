import numpy as np
from rand import ensure_rng

class Target(object):
    """
    """
    def __init__(self, func, pbounds, random_state=None):
        """
        Paramepers:
        -----------
        target_func: 
        pbounds: dict
            Dictionary with parameters names as keys and 
            a tuple with minimum and maximum array or list values
        """

        self.func = func
        self.keys = list(pbounds.keys())
        self.pbounds = pbounds  
        self.random_state = ensure_rng(random_state=random_state)

        self.xdim=0
        for key, (lower, upper) in self.pbounds.items():
            self.xdim+=lower.shape[0]

    def set_random_state(self,random_state): 
        self.random_state = ensure_rng(random_state=random_state)

    def random_points(self, num):
        """
        Parameters:
        ------------

        Returns:
        ------------

        """
        X = {}
        for key, (lower, upper) in self.pbounds.items():
            assert type(lower)==type(upper) and type(lower)==np.ndarray
            assert lower.shape==upper.shape
            shape = (num,)+lower.shape
            rand = self.random_state.uniform(0,1,shape)
            X[key] = rand*(upper-lower)+lower
        
        return X


    def order_points(self, num_per_dim):
        """
        Parameters:
        ------------

        Returns:
        ------------

        """
        X = {}
        for key, (lower, upper) in self.pbounds.items():
            assert type(lower)==type(upper) and type(lower)==np.ndarray
            assert lower.shape==upper.shape
            dim = lower.shape[0]
            rang = np.ones(dim)*np.array([np.linspace(0,1,num_per_dim)]).T
            rang = rang*(upper-lower)+lower
            X[key] = rang
            #xi = [rang[:,i] for i in range(dim)]
            #X[key] =  np.stack(np.meshgrid(*xi), axis=dim).reshape((-1,dim))

        return X


    def set_bounds(self, new_pbounds):
        """
        Parameters:
        ------------
        new_bounds: dict

        """
        for row, key in enumerate(self.keys):
            if key in new_pbounds:
                self.pbounds[key] = new_pbounds[key]
    

    def get_output(self, **params):
        return self.func(**params)
    

    def dict_to_array(self, **params):
        """
        dict tranpose to array
        """
        return np.concatenate([params[key] for key in self.keys])




