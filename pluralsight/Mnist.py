import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32,shape=[None,784])#image 28x28
y_ = tf.placeholder(tf.float32,shape=[None,10])# real
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)# y => tinh toan, y_ => real

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)) #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs,batch_ys = mnist.train.next_batch(100)
  sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#after tranning, the next step is testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
test_accuracy = sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print("Test Accuracy: {0}%".format(test_accuracy *100.0))

sess.close()

