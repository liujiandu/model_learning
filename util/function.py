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
    Retrun:
        [sample_num, 1]
    """    
    #z = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
    z=x
    return np.mean(z, axis=1).reshape((-1,1))

