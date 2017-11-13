import tensorflow as tf
import numpy as np
tf.reset_default_graph()

w1 = tf.Variable(initial_value=2, name="PleaseUpdateIt", dtype=np.float64)
w2 = tf.Variable(initial_value=4, name="PleaseDontUpdateItButTakeItIntoAccount", dtype=np.float64)
whatWeWantToUpdate = [w1,] #this is the important line. change this list and you will see that the error changes.
#TA confirmed: w2 will be taken into account but not updated :D

x = tf.placeholder(dtype = np.float64, name="x")
f = tf.multiply(w1, x)
yhat = tf.multiply(w2, f)
E = tf.multiply(np.float64(.5), tf.square(tf.subtract( yhat, 16 ))) # E = 0.5*(yhat-16)^2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
gradients = optimizer.compute_gradients(loss=E, var_list=whatWeWantToUpdate)
train_op = optimizer.apply_gradients(gradients)
with tf.Session() as sess:
    eat_this = {x:1}
    sess.run(tf.global_variables_initializer())
    print("the error at the beginning: ", sess.run(E, eat_this))
    print("the gradients: ", sess.run(fetches=gradients, feed_dict=eat_this) ) 
    sess.run(fetches=train_op, feed_dict=eat_this)
    print("the error after one training run: ", sess.run(E, eat_this))
