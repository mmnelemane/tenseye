
import tensorflow as tf
print(tf.version)

t = tf.zeros([5,5,5,5])
print(t)

# Flattening (reshape)

t = tf.reshape(t, [625])
print(t)

