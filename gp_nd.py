#=========================
#gaussian process fit 2d function
#Author: Liujiandu
#Date: 2018/1/9
#=========================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mpl_toolkits.mplot3d import Axes3D

def target(x):
    """
    target function n dimension
    :Parma x:
        [sample_num, feature_dim]
    """    
    z=0
    n=x.shape[1]
    for i in range(n):

        #z += np.exp(-(x[:,i]-2)**2)+np.exp(-(x[:,i]-6)**2/5)+1/(x[:,i]**2+1)+0.1*np.sin(5*x[:,i])-0.5
        z+=x[:,i]
    return z/n

def gp_fit(X, Y, x):
    """
    :Param X:
        input sampled data points [samples_num, feature_dim]
    :Param Y:
        output sampled data points [samples_num, [y_dim]]
    :Param x:
        all input data points [points_num, feature_dim]
    """
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),n_restarts_optimizer=25)
    gp.fit(X, Y)
    mu, sigma = gp.predict(x, return_std=True)
    return mu, sigma    
    
    
def eval(func, points_num, bounds, x_dim):
    X=[]
    for i in range(x_dim):
        Xi = (np.random.random((points_num, 1))-0.5)*(bounds[1][i]-bounds[0][i])+(bounds[1][i]+bounds[0][i])/2.0
        X.append(Xi)
    X = np.concatenate(X, axis=1) #[points_num, n]
    Y = func(X) #[points_num,]
    
    #calculate x space
    x=[]
    for i in range(x_dim):
        xi = (np.random.random((100000, 1))-0.5)*(bounds[1][i]-bounds[0][i])+(bounds[1][i]+bounds[0][i])/2.0
        x.append(xi)
    x = np.concatenate(x, axis=1) #[points_num, n]
    #calculate y
    y = func(x) 
    #calculate mu, sigma
    mu, sigma = gp_fit(X, Y, x)
    #cal mse
    mse = np.sum(np.square(y-mu))

    return mse



def number_mse(x_dim, num):
    """
    :Parma x_dim:
        dimension of x
    """
    mses = []
    bounds=[-5*np.ones(x_dim), 10*np.ones(x_dim)]
    for i in range(1,num):
        mse = eval(target, 100*i, bounds, x_dim)
        mses.append(mse)
    print mses
    plt.plot(range(1,num), mses, linewidth=2)
    plt.plot(range(1,num), mses, "D", markersize=6)
    plt.xlabel('sampled points number')
    plt.ylabel('mean square error')
    plt.grid(True)
    plt.show()

def xdim_mse(num_points, num):
    """
    :Param num_points:
        number of sampled data points
    :Param num:
        repeat times
    """
    mses = []
    for x_dim in range(1,num):
        bounds=[-5*np.ones(x_dim), 10*np.ones(x_dim)]
        mse = eval(target, num_points, bounds, x_dim)
        mses.append(mse)
    print mses
    plt.plot(range(1,num), mses, linewidth=2)
    plt.plot(range(1,num), mses, "D", markersize=6)
    plt.xlabel('x dimension')
    plt.ylabel('mean square error')
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    number_mse(2,5)
    #xdim_mse(50, 10)
