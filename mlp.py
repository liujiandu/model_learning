#=====================
# multilayer neural network
# Author: Liujiandu
# Date: 2018/1/10
#=====================
import tensorflow as tf

def fc(x, shape, scope, actf=tf.nn.tanh):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [shape[1]], initializer=tf.constant_initializer(0.01))
        z = actf(tf.matmul(x,w)+b)
    return z


#network
class MLP(object):
    def __init__(self, x_dim, y_dim, hidden_dim, scope='mlp', reuse=False):
        """
        :Param hidden_dim:
            hidden layer neuron number, list like
        """
        self.hidden_dim=hidden_dim
        self.x = tf.placeholder(tf.float32, [None, x_dim])
        self.y_ = tf.placeholder(tf.float32,[None, y_dim])
        
        with tf.variable_scope(scope, reuse=reuse):
            f = fc(self.x, [x_dim, self.hidden_dim[0]], 'fc1',actf=tf.nn.tanh)
            for i in range(len(self.hidden_dim)-1):
                f = fc(f, [self.hidden_dim[i], self.hidden_dim[i+1]], 'fc'+str(i),actf=tf.nn.tanh)
            self.y = fc(f, [self.hidden_dim[-1], y_dim], 'y', actf=tf.identity)

            self.loss = tf.reduce_sum(tf.reduce_sum(tf.square(self.y-self.y_),reduction_indices = [1]))

            #train
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        #initialize
        #init = tf.global_variables_initializer()
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def update(self, x, y_): 
        self.sess.run(self.train_step, feed_dict=({self.x:x, self.y_:y_}))
    def get_y(self, x):    
        return self.sess.run(self.y, feed_dict={self.x:x})

    def get_loss(self, x, y_):    
        return self.sess.run(self.loss, feed_dict={self.x:x, self.y_: y_})

def eval_mlp(mlp, func, xs, x, max_iter=1000):
    if xs.shape[1] != mlp.x.get_shape()[1]:
        return 0
    ys = func(xs)
    if ys.shape[1] != mlp.y_.get_shape()[1]:
        return 0
    #model regression
    for i in range(max_iter):
        batch_xs, batch_ys=(xs,ys)
        mlp.update(batch_xs, batch_ys)
        if i%100==0: 
            loss=mlp.get_loss(xs,ys.reshape((-1,1)))
            print loss
            if loss<1e-4:
                break

    
    #prediction
    y = mlp.get_y(x)
    return np.sum(np.square(y-func(x)))

if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from util.function import func
    
    mses=[]
    for i in range(1,2):
        #mlp
        x_dim=3
        y_dim=1
        hidden_dim=[50,20]
        mlp = MLP(x_dim, y_dim, hidden_dim, 'mlp'+str(i))
        
        #xs, x
        xs = (np.random.random((1000*i,x_dim))-0.5)*15+2.5
        x = (np.random.random((100000,x_dim))-0.5)*15+2.5
        
        #eval
        mse = eval_mlp(mlp, func, xs, x, 20000)
        mses.append(mse)
    print mses
    fig = plt.figure()
    plt.plot(range(1,10), mses, 'D', markersize=6, color='r')
    plt.plot(range(1,10), mses, linewidth=2)
    plt.grid(True)
    plt.show()
    
