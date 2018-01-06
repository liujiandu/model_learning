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
    target function 2 dimension
    """
    z = np.exp(-(x[:,0]-2)**2)+np.exp(-(x[:,0]-6)**2/5)+1/(x[:,0]**2+1)+0.1*np.sin(5*x[:,0])-0.5
    z += np.exp(-(x[:,1]-2)**2)+np.exp(-(x[:,1]-6)**2/5)+1/(x[:,1]**2+1)+0.1*np.sin(5*x[:,1])-0.5
    return z/2

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
    
def plot_gp(X, Y, x, y, mu):  
    """
    :Param X:
        [num_sample, 2]
    :Param Y:
        [num_sample, 1]
    :Param x:
        [column_dim, row_dim, 2 ]
    :Param y:
        [column_dim, row_dim]
    :Param mu:
        [column_dim, row_dim]
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x[:,:,0],x[:,:,1],y,rstride=1, cstride=1, cmap='rainbow', alpha=1)
    
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    #ax1.scatter(X[:,0], X[:,1], Y,s=40, c='r')
    ax1.plot_surface(x[:,:,0],x[:,:,1],mu,rstride=1, cstride=1, cmap='rainbow', alpha=1)
    plt.show()
    
def eval(func, points_num, bounds, display=False):
    X1 = (np.random.random((points_num, 1))-0.5)*(bounds[1][0]-bounds[0][0])+(bounds[1][0]+bounds[0][0])/2.0
    X2 = (np.random.random((points_num, 1))-0.5)*(bounds[1][1]-bounds[0][1])+(bounds[1][1]+bounds[0][1])/2.0
    X = np.concatenate((X1, X2), axis=1) #[points_num, 2]
    Y = func(X) #[points_num,]
    
    #calculate x space
    a = np.arange(bounds[0][0], bounds[1][0], 0.2)
    b = np.arange(bounds[0][1], bounds[1][1], 0.2)
    a,b = np.meshgrid(a,b) 
    x = np.stack((a,b), axis=2) #[a_dim, b_dim, 2]
    
    #calculate y
    y = func(x.reshape((-1,2))) #[a_dim*b_dim]
    
    #calculate mu, sigma
    mu, sigma = gp_fit(X, Y, x.reshape((-1,2))) #[a_dim*b_dim]
    
    #cal mse
    mse = np.sum(np.square(y-mu))
    
    mu = mu.reshape(a.shape) #[a_dim, b_dim]
    y = y.reshape(a.shape) #[a_dim, b_dim]
    
    if display:
        plot_gp(X,Y, x, y, mu)
    return mse

if __name__=="__main__":
    mses = []
    for i in range(1,50):
        mse = eval(target, 10*i,[[-5, -5],[10,10]], display=False)
        mses.append(mse)
    plt.plot(range(1,50), mses, linewidth=3)
    plt.plot(range(1,50), mses, "D", markersize=8)
    plt.show()
