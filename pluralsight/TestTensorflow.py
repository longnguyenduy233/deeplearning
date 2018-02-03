import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello ...")
print(sess.run(hello))

a = tf.constant(20)
b = tf.constant(50)

print("A + B = {0}".format(sess.run(a+b)))