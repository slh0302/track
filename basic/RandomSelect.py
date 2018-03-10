import tensorflow as tf

def randomSelectIndex(fromCount, n):
	with tf.name_scope("randomSelectIndex"):
		n = tf.minimum(fromCount, n)
		i = tf.random_shuffle(tf.range(fromCount, dtype=tf.int32))[0:n]
		return tf.expand_dims(i,-1)

def randomSelectBatch(t, n):
	with tf.name_scope("randomSelectBatch"):
		count = tf.shape(t)[0]
		return tf.gather_nd(t, randomSelectIndex(count,n))