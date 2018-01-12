#========================
#gaussian process fit 2d function
#Author: Liujiandu
#Date: 2018/1/9
#=========================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D


def plot_curve(X, Y, x, y, mu=None, sigma=None):
    """
    Parameters:
    :X: 2d-array
        input sampled data points
    :Y: 2d-array
        output sampled data points
    :x: 2d-array
        all input data points
    :y: 2d-array
        all output data points
    :mu: 2d-array
        mean of predicted gaussain process
    :sigma: 2d-array
        standard variance of precidted gaussian process
    """

    fig = plt.figure(figsize=(16,10))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    axis = plt.subplot(gs[0])
    #acq = plt.subplot(gs[1])
    
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(X, Y, 'D', markersize=8, color='r', label='Observation')
    if mu is not None:
        axis.plot(x, mu, '--', color='k', label='Prediction')
        if sigma is not None:
            axis.fill(np.concatenate([x, x[::-1]]), np.concatenate([mu-1.96*sigma, (mu+1.96*sigma)[::-1]]), alpha=0.6, fc='c', ec='None')

    plt.show()


def plot_surface(x, y, mu):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x[:,:,0], x[:,:,1], y, rstride=1, cstride=1, cmap='rainbow', alpha=1)

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.plot_surface(x[:,:,0], x[:,:,1], mu, rstride=1, cstride=1, cmap='rainbow', alpha=1)
    plt.show()


def eval(target, regressor, points_num, display=False):  
    """
    Parmeters:
    -----------
    target:
    regressor:
    points_num:
    display:
    """
    #calculate sample points
    X = target.random_points(points_num)
    Y = target.get_output(**X) #[points_num,]
        
    #calculate x space
    x = target.order_points(50)
    y = target.get_output(**x) 

    #calculate mu, sigma
    regressor.fit(target.dict_to_array(**X),Y)
    mu, sigma = regressor.predict(target.dict_to_array(**x))
    mse = np.sum(np.mean(np.square(y-mu), axis=0))
    
    #plot
    if display:
        if target.xdim==1:
            plot_curve(target.dict_to_array(**X), Y, target.dict_to_array(**x), y, mu=mu.flatten(), sigma=sigma)
        if target.xdim==2:
            plot_surface(target.dict_to_array(**x).reshape((100,100,2)), y.reshape((100,100)), mu.reshape((100,100)))
    return mse



def pointnum_mse(target, regressor, itera, display=False):
    """
    Parameters:
    -------------
    target:

    num:

    """
    mses = []

    for i in range(1, itera):
        points_num = i*10
        mse = eval(target, regressor, points_num, display=False)
        print mse
        mses.append(mse)
    print mses
    
    if display:
        plt.plot(range(1,itera), mses, linewidth=2)
        plt.plot(range(1,itera), mses, "D", markersize=6)
        plt.xlabel('sampled points number')
        plt.ylabel('mean square error')
        plt.grid(True)
        plt.savefig('num_mse_gp.png')
        plt.show()

    return mses

'''
def xdim_mse(target, points_num, itera):
    """
    :Param num_points:
        number of sampled data points
    :Param num:
        repeat times
    """
    mses = []
    for x_dim in range(1,itera):
        mse = eval(target, num_points)
        print mse
        mses.append(mse)
    print mses

    plt.plot(range(1,num), mses, linewidth=2)
    plt.plot(range(1,num), mses, "D", markersize=6)
    plt.xlabel('x dimension')
    plt.ylabel('mean square error')
    plt.grid(True)
    plt.savefig('xdim_mse_gp.png')
    plt.show()
'''



if __name__=="__main__":
    from util.function import nlfunc, lfunc
    from util.target import Target
    from regressor.gp import GPR
    from regressor.mlp import MLP
    
    ##target function
    x_dim = 3
    random_state = 1

    bounds = {'x':(-5*np.ones(x_dim), 10*np.ones(x_dim))}
    target = Target(nlfunc, bounds, random_state=random_state)
    
    #=========GPR==================
    regressor = GPR(kername='matern',n_restarts_optimizer=25)
    #print eval(target, regressor, 20, display=False)
    mse_gp = pointnum_mse(target, regressor, 10, display=False)
    
    
    #============MLP=============
    regressor = MLP(x_dim,1, [50, 30])
    #print eval(target, regressor, 20, display=False)
     
    target.set_random_state(random_state)
    mse_mlp = pointnum_mse(target, regressor, 10, display=False)
    mse_mlp = np.array(mse_mlp)
    for _ in range(5):
        target.set_random_state(random_state)
        mse_mlp1 = pointnum_mse(target, regressor, 10, display=False)
        mse_mlp +=  np.array(mse_mlp1)
    mse_mlp /=5

    #=========plot===============
    plt.plot(np.arange(1,10,1)*10, mse_gp, label='gp', linewidth=2)
    plt.plot(np.arange(1,10,1)*10, mse_mlp, label='mlp 5_ave', linewidth=2)
    plt.legend()
    plt.xlabel('points number')
    plt.ylabel('mean square error')
    plt.grid(True)
    plt.savefig('./result/mlp_gp_com_{}.png'.format(x_dim))
    plt.show()
    



