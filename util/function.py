#=========================
#gaussian process fit 2d function
#Author: Liujiandu
#Date: 2018/1/9
#=========================
import numpy as np

def func(x):
    """
    target function n dimension
    :Parma x:
        [sample_num, feature_dim]
    """    
    z=0
    n=x.shape[1]
    for i in range(n):
        z += np.exp(-(x[:,i]-2)**2)+np.exp(-(x[:,i]-6)**2/5)+1/(x[:,i]**2+1)+0.1*np.sin(5*x[:,i])-0.5
    return z/n

