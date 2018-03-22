from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.MultiLoader.MultiBoxLoader import MultiBoxLoader
from MultiModel.MultiLoader.MultiCar import MultiCarDataset
from datasets.process import Augment
from tensorflow.python.ops import control_flow_ops
from utils.CheckpointLoader import loadCheckpoint
import tensorflow as tf
import sys
import os


log_dir = "logSingleRFCN0/"
model_dir = "modelSingleRFCN0/"

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
learning_rate = 0.001
batch = 8
globalStep = tf.Variable(0, name='globalStep', trainable=False)
globalStepInc = tf.assign_add(globalStep, 1)
print("learning: ", learning_rate)
# initial_learning_rate = 0.1

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

        update_ops.append(optimizer.apply_gradients(grads))
        return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

# load data
dataset = MultiBoxLoader(batch=batch)
dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch))
# dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch))
images, boxes, classes, number = Augment.MutliAugment(*dataset.get())

num_epoch_for_decay = 2
# call to calc
dataset.init()
decay_steps = int((dataset.count() / (batch * 2 * 1)) * num_epoch_for_decay)
learning_rate = tf.train.exponential_decay(learning_rate,
                                           global_step=globalStep,
                                           decay_steps=decay_steps, decay_rate=0.8, staircase=True)

# load module
net = MultiBoxInceptionResnet(images, 1, name="boxnet", hardMining=True, batch=batch, trainFrom=0)
loss = net.getLoss(boxes, classes, number)
tf.losses.add_loss(loss)
trainOp = createUpdateOp(net)

# summary
tf.summary.scalar("loss_t", loss)
tf.summary.scalar("learning_1", learning_rate)

# run phases
# config.gpu_options.allow_growth = True
saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)
config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("../logs/" + log_dir, sess.graph)

    print("Loading GoogleNet")
    sess.run(tf.global_variables_initializer())


    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
    # print("Done.")

    if not loadCheckpoint(sess, None, "../../save/model_3/model", ignoreVarsInFileNotInSess=True, ignoreClass=True):
        print("reload network.")
        sys.exit(-1)

    dataset.startThreads(sess)

    i = 1
    cycleCnt = 0
    lossSum = 0

    while True:
        try:
            i, ls, tbox, nm, rpnls, bxls= sess.run([globalStepInc, trainOp, net.totalBox, net.testnumber, net.rpnloss, net.boxLoss])
        except KeyboardInterrupt:
            print("Keyboard interrupt. Shutting down")
            sys.exit(0)

        lossSum += ls
        cycleCnt += 1
        # average = 0
        print("setp ",i," loss: ", ls, " testnumber: ", nm.tolist(), " toal box:",[tbox[i].shape for i in range(batch)],
              " rpn loss:", rpnls, " boxls:", bxls)

        if i % 50 == 0:
            if cycleCnt > 0:
                loss = lossSum / cycleCnt

            # lossS=sess.run(trainLossSum, feed_dict={
            # 	trainLossFeed: loss
            # })
            # log.add_summary(lossS, global_step=samplesSeen)

            epoch = "%.2f" % (float(i * batch * 2) / dataset.count())
            print("Iteration " + str(i) + " (epoch: " + epoch + "): loss: " + str(loss))
            lossSum = 0
            cycleCnt = 0

        if i % 200 == 0:
            # average = lossSum / cycleCnt
            print("Saving checkpoint " + str(i))
            saver.save(sess, "../model/"+ model_dir + "model_" + str(i*batch*2), write_meta_graph=False)

        # tf.summary.scalar("loss_aver", average)
        merged_summary_op = tf.summary.merge_all()
        summary_merged = sess.run(merged_summary_op)
        writer.add_summary(summary_merged, i)
