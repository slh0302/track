import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from model.RFCN import RFCN
from utils.BoxLoader import *
from datasets.car import CarDataset
from datasets.process import Augment

globalStep = tf.Variable(0, name='globalStep', trainable=False)
globalStepInc=tf.assign_add(globalStep,1)

# load data
dataset = BoxLoader()
dataset.add(CarDataset("/home/slh/dataset/DETRAC/", randomZoom=True, set="Train"))
images, boxes, classes = Augment.augment(*dataset.get())

# load module
basenet = RFCN(dataset, 4, 50, hardMining=False)
# opt


# run phases
saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)) as sess:
    print("Loading GoogleNet")
    basenet.importWeights(sess, "./inception_resnet_v2_2016_08_30.ckpt")
    # net.importWeights(sess, "initialWeights/", permutateRgb=False)
    print("Done.")
    dataset.startThreads(sess)
    pass

# def createUpdateOp(gradClip=1):
# 	with tf.name_scope("optimizer"):
# 		optimizer=tf.train.AdamOptimizer(learning_rate=opt.learningRate, epsilon=opt.adamEps)
# 		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# 		totalLoss = tf.losses.get_total_loss()
# 		grads = optimizer.compute_gradients(totalLoss, var_list=net.getVariables())
# 		if gradClip is not None:
# 			cGrads = []
# 			for g, v in grads:
# 				if g is None:
# 					print("WARNING: no grad for variable "+v.op.name)
# 					continue
# 				cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
# 			grads = cGrads
#
# 		update_ops.append(optimizer.apply_gradients(grads))
# 		return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')
