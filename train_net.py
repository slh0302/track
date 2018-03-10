import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def createUpdateOp(gradClip=1):
	with tf.name_scope("optimizer"):
		optimizer=tf.train.AdamOptimizer(learning_rate=opt.learningRate, epsilon=opt.adamEps)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		totalLoss = tf.losses.get_total_loss()
		grads = optimizer.compute_gradients(totalLoss, var_list=net.getVariables())
		if gradClip is not None:
			cGrads = []
			for g, v in grads:
				if g is None:
					print("WARNING: no grad for variable "+v.op.name)
					continue
				cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
			grads = cGrads

		update_ops.append(optimizer.apply_gradients(grads))
		return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')
