from nModel.RPN import RPN
from nModel.BoxInceptionResnet import BoxInceptionResnet
from nModel.Loader.BoxLoader import BoxLoader
from nModel.Loader.car import CarDataset
from datasets.process import Augment
from tensorflow.python.ops import control_flow_ops
from utils.CheckpointLoader import loadCheckpoint
import tensorflow as tf
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
learning_rate = 0.001
globalStep = tf.Variable(0, name='globalStep', trainable=False)
# globalStepInc = tf.assign_add(globalStep, 1)
print("learning: ", learning_rate)
# initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=globalStep,
                                           decay_steps=80000, decay_rate=0.9)
def createUpdateOp(net, gradClip=1):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        totalLoss = tf.losses.get_total_loss()
        grads = optimizer.compute_gradients(totalLoss, var_list=net.getVariables())
        if gradClip is not None:
            cGrads = []
            for g, v in grads:
                if g is None:
                    print("WARNING: no grad for variable " + v.op.name)
                    continue
                cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
            grads = cGrads

        update_ops.append(optimizer.apply_gradients(grads, global_step=globalStep))
        return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

# load data
dataset = BoxLoader()
dataset.add(CarDataset("/home/slh/dataset/DETRAC/", set="Train"))
# images, boxes, classes, boxNum, clsNum, refdisp, dispLen = Augment.augmentImages(*dataset.get())
images, boxes, classes = Augment.augment(*dataset.get())
# load module
net = BoxInceptionResnet(images, 4, name="boxnet", hardMining=True, trainFrom=0)
loss = net.getLoss(boxes, classes)
tf.losses.add_loss(loss)
trainOp = createUpdateOp(net)
tf.summary.scalar("loss_4", loss)
# merged_summary_op = tf.summary.merge_all()
# run phases

config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("../logs_same/", sess.graph)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)
    print("Loading GoogleNet")

    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
    if not loadCheckpoint(sess, None, "../../save/model_3/model", ignoreVarsInFileNotInSess=True, ignoreClass=True):
       print("reload network.")
       sys.exit(-1)

    print("Done.")

    i = sess.run(globalStep)
    dataset.startThreads(sess)
    sess.graph.finalize()

    cycleCnt = 0
    lossSum = 0

    while True:
        try:
            i, ls, rpnls, bxls = sess.run([globalStep, trainOp, net.rpnloss, net.boxLoss])
        except KeyboardInterrupt:
            print("Keyboard interrupt. Shutting down.")
            sys.exit(0)

        print("setp ", i, " loss: ", ls,
              " rpn loss:", rpnls, " boxls:", bxls)
        lossSum += ls
        cycleCnt += 1
        # average = 0
        if i % 50 == 0:
            if cycleCnt > 0:
                loss = lossSum / cycleCnt

            epoch = "%.2f" % (float(i) / dataset.count())
            print("Iteration " + str(i) + " (epoch: " + epoch + "): loss: " + str(loss))
            lossSum = 0
            cycleCnt = 0

        if i % 2000 == 0:
            # average = lossSum / cycleCnt
            print("Saving checkpoint " + str(i))
            saver.save(sess, "../model/model_same/model_" + str(i), write_meta_graph=False)

        # tf.summary.scalar("loss_aver", average)