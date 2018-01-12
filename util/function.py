"""
test function for Optimization algorithm,
or neural network, or other model

Author: Liujiandu
Date: 2018/1/9
"""

import numpy as np


def nlfunc(x):
    """
    target function is a multi-dimensional nonlinear function 

    Parameters:
    -----------
    x: ndarray, ndim>=2
       input of function
    
    Retruns:
    ----------
    y: ndarray  
       output of function
    """    
    z = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
    y = np.mean(z, axis=-1)   
    return y.reshape(y.shape+(1,))

def lfunc():
    """
    target function is a multi-dimensional nonlinear function 

    Parameters:
    -----------
    x: ndarray, ndim>=2
       input of function
    
    Retruns:
    ----------
    y: ndarray,
       output of function
    """
    z = x
    y = np.mean(z, axis=1)
    return y.reshape(y.shape+(1,))



if __name__=="__main__":
    x = np.array([1,2,3,4])
    print x.shape

    x={"x":x}
    y = nlfunc(**x)
    print y.shape
