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
    def __init__(self, x_dim, y_dim, hidden_dim):
        """
        :Param hidden_dim:
            hidden layer neuron number, list like
        """
        self.hidden_dim=hidden_dim
        self.x = tf.placeholder(tf.float32, [None, x_dim])
        self.y_ = tf.placeholder(tf.float32,[None, y_dim])

        f = fc(self.x, [x_dim, self.hidden_dim[0]], 'fc1',actf=tf.nn.tanh)
        for i in range(len(self.hidden_dim)-1):
            f = fc(f, [self.hidden_dim[i], self.hidden_dim[i+1]], 'fc'+str(i),actf=tf.nn.tanh)
        self.y = fc(f, [self.hidden_dim[-1], y_dim], 'y', actf=tf.identify)

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y-self.y_),reduction_indices = [1]))

        #train
        self.train_step = tf.train.AdagradOptimizer(0.3).minimize(self.loss)

        #initialize
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess(init)


if __name__=="__main__":
    from util.function import func
    x_dim=2
    y_dim=1
    mlp = MLP(x_dim, y_dim, [20, 10])
    xs = np.random.random(100,2)
    ys = util.func(xs)
    for _ in range(1000):
        batch_xs, batch_ys=(xs,ys)
        sess.run(train_step, feed_dict=({x:batch_xs, y_:batch_ys}))
    print(sess.run(loss, feed_dict=({x:xs, y_:ys})))

