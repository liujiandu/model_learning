#=========================
#Gaussian process fit one dimension function
#Author: Liujiandu
#Date: 2018/1/8
#=========================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def target(x):
    """
    target funtion 1 dimension
    """
    y = np.exp(-(x-2)**2)+np.exp(-(x-6)**2/5)+1/(x**2+1)+0.1*np.sin(5*x)-0.5
    return y*10    

def fit_gp(X, Y, x,y):
    """
    gaussian process regressor
    :param X:
        input of sampled data points 
    
    :param Y:
        output of sampled data points

    :param x:
        all input of data points
    
    :param y:
        all output of data points

    Return:
        mu: mean of output
        sigma : standard variance of output
    """
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5),n_restarts_optimizer=25)
    gp.fit(X, Y)
    mu, sigma = gp.predict(x, return_std=True)
    
    return mu, sigma

def plot_gp(X, Y, x, y, mu, sigma):
    """
    :param X:
        input sampled data points
    :param Y:
        output sampled data points
    :parma x:
        all input data points
    :parma y:
        all output data points
    :param mu:
        mean of predicted gaussain process
    :param sigma:
        standard variance of precidted gaussian process
    
    """
    fig = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(X.flatten(), Y, 'D', markersize=8, color='r', label='Observation')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

    plt.show()



def eval(func, points_num, bounds, display=False):
    """
    evaluate model fitted by gaussian process

    :param func:
        target function
    :param points_num:
        the number of sampled points
    :param bounds:
        bounds of input range, 1d array or list [min, max]
    
    Return:
        mean square error between mu and y     
    """
    X = ((np.random.random(points_num)-0.5)*(bounds[1]-bounds[0])+(bounds[0]+bounds[1])/2).reshape(-1,1)
    Y = func(X)
    x = np.linspace(bounds[0], bounds[1], 500).reshape(-1,1)
    y = func(x)
    mu, sigma = fit_gp(X, Y, x, y)
    if display:
        plot_gp(X, Y, x, y, mu, sigma)

    return np.sum(np.square(mu.flatten()-y.flatten()))


if __name__=="__main__":
    mses = []
    for i in range(1,15):
        if i==19:
            display=True
        else:
            display=False

        mse = eval(target, i*10, [-5.0, 10.0], display=display)
        mses.append(mse)
    print mses
    plt.plot(range(1,15), mses, linewidth=3, color='b')
    plt.plot(range(1,15), mses, 'D', markersize=8, color='r')
    plt.show()

