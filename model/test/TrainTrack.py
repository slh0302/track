from model.TrackRPN import TrackRPN
from model.InceptionRFCN import InceptionRFCN
from utils.TrackBoxLoader import TrackBoxLoader
from datasets.car import CarDataset
from datasets.process import Augment
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
import sys
import os
# 3 0.01
# 2 0.1
# 4 0.001
# 5 0.001 decay
os.environ["CUDA_VISIBLE_DEVICES"] = "13"
learning_rate = 0.001
globalStep = tf.Variable(0, name='globalStep', trainable=False)
globalStepInc = tf.assign_add(globalStep, 1)
print("learning: ", learning_rate)
# initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=globalStep,
                                           decay_steps=800, decay_rate=0.6)
def createUpdateOp(net, gradClip=1):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
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

        update_ops.append(optimizer.apply_gradients(grads))
        return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

# load data
dataset = TrackBoxLoader()
dataset.add(CarDataset("/home/slh/dataset/DETRAC/", set="Train"))
# images, boxes, classes, boxNum, clsNum, refdisp, dispLen = Augment.augmentImages(*dataset.get())
images, boxes, classes, refdisp = Augment.augmentImages(*dataset.get())
# load module
basenet = InceptionRFCN(images, 4, base_model=TrackRPN, hardMining=False, trainFrom=0)
loss = basenet.getLoss(boxes, classes, refdisp)
tf.losses.add_loss(loss)
trainOp = createUpdateOp(basenet)
tf.summary.scalar("loss_5", loss)
merged_summary_op = tf.summary.merge_all()
# run phases
saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)) as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)

    print("Loading GoogleNet")
    sess.run(tf.global_variables_initializer())
    basenet.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
    # net.importWeights(sess, "initialWeights/", permutateRgb=False)
    print("Done.")
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    # for k, v in zip(variable_names, values):
    #     print("Variable: ", k)
    #     print("Shape: ", v.shape)
        # print(v)

    dataset.startThreads(sess)

    i = 1
    cycleCnt = 0
    lossSum = 0

    while True:
        try:
            i, ls, posBox= sess.run([globalStepInc, trainOp, basenet.Rpn.Corr.posboxNum])
        except KeyboardInterrupt:
            print("Keyboard interrupt. Shutting down.")
            sys.exit(0)
        print("box:  ", posBox)
        lossSum += ls
        cycleCnt += 1
        if i % 10 == 0:
            if cycleCnt > 0:
                loss = lossSum / cycleCnt

            # lossS=sess.run(trainLossSum, feed_dict={
            # 	trainLossFeed: loss
            # })
            # log.add_summary(lossS, global_step=samplesSeen)

            epoch = "%.2f" % (float(i) / dataset.count())
            print("Iteration " + str(i) + " (epoch: " + epoch + "): loss: " + str(loss))
            lossSum = 0
            cycleCnt = 0

        if i % 200 == 0:
            print("Saving checkpoint " + str(i))
            saver.save(sess, "../../save/model_" + str(i), write_meta_graph=False)

        summary_merged = sess.run(merged_summary_op)
        writer.add_summary(summary_merged, i)