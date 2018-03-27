from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.MultiLoader.MultiBoxLoader import MultiBoxLoader
from MultiModel.MultiLoader.MultiCar import MultiCarDataset
from datasets.process import Augment
from memory_profiler import profile
from tensorflow.python.ops import control_flow_ops
from utils.CheckpointLoader import loadCheckpoint
import tensorflow as tf
import sys
import os

log_dir = "single/reOld9_2/"
model_dir = "single/reOld9_2/"
NumClasses = 1
trainFrom=7
learning_rate = 0.001
batch = 4
LoadOld=True
ignoreClass=True
staircase=True
opts="sgd"
modelToLoad = "../../save/model_3/model" #"/home/slh/tf-project/track/MultiModel/model/single/oldEnd2Train/model_60000"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def test_Block():
#


# graph
with tf.Graph().as_default() as graph:
    globalStep = tf.Variable(0, name='globalStep', trainable=False)
    # globalStepInc = tf.assign_add(globalStep, 1)
    print("learning: ", learning_rate)
    # initial_learning_rate = 0.1
    # load data
    dataset = MultiBoxLoader(batch=batch)
    dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch, Noc=NumClasses))
    # dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch))
    images, boxes, classes, number = dataset.get()

    images, boxes, classes, number = Augment.MutliAugment(images, boxes, classes, number)

    num_epoch_for_decay = 4
    # call to calc
    dataset.init()
    decay_steps = int((dataset.count() / (batch * 2 * 1)) * num_epoch_for_decay)
    learning_rate = tf.train.exponential_decay(learning_rate,
                                               global_step=globalStep,
                                               decay_steps=decay_steps, decay_rate=0.9, staircase=staircase)

    def createUpdateOp(net, gradClip=1):
        with tf.name_scope("optimizer"):
            if opts == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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

    # load module
    net = MultiBoxInceptionResnet(images, NumClasses, name="boxnet", hardMining=True, batch=batch, trainFrom=trainFrom)
    loss, _ = net.getLoss(boxes, classes, number)
    tf.losses.add_loss(loss)
    trainOp = createUpdateOp(net)

    # summary
    tf.summary.scalar("loss_t_1", loss)
    tf.summary.scalar("learning_2", learning_rate)

# run phases
# config.gpu_options.allow_growth = True
config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=graph) as sess:
    # begin
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=100)
    if not os.path.exists("../logs/" + log_dir):
        os.mkdir("../logs/" + log_dir)
    writer = tf.summary.FileWriter("../logs/" + log_dir, sess.graph)

    print("Loading GoogleNet")
    sess.run(tf.global_variables_initializer())

    if LoadOld:
        if not loadCheckpoint(sess, None, modelToLoad, ignoreVarsInFileNotInSess=True, ignoreClass=ignoreClass):
            print("reload network.")
            sys.exit(-1)
    else:
        net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
        print("Done.")


    dataset.startThreads(sess)
    shape1 = sess.run(tf.shape(net.scale_32_2))
    #
    # hIn, wIn = sess.run([n?et.rpn.hIn, net.rpn.wIn])
    # tf.summary.scalar("loss_aver", average)
    # make graph read only

    i = sess.run(globalStep)

    # sess.graph.finalize()
    results = []
    cycleCnt = 0
    lossSum = 0
    while True:
        try:
            i, ls, tbox, rpnls, bxls, testNumber= sess.run([globalStep, trainOp, net.totalBox, net.rpnloss, net.boxLoss, net.testnumber])
        except KeyboardInterrupt:
            sess.close()
            writer.close()
            del sess
            print("Keyboard interrupt. Shutting down")
            sys.exit(0)

        lossSum += ls
        cycleCnt += 1
        # average = 0
        print("setp ",i," loss: ", ls,
              " rpn loss:", rpnls, " boxls:", bxls, " nm:", testNumber.tolist())
        # print("setp ",i," loss: ",  ls)

        if i % 50 == 0:
            if cycleCnt > 0:
                loss = lossSum / cycleCnt

            epoch = "%.2f" % (float(i * batch * 2) / dataset.count())
            print("Iteration " + str(i) + " (epoch: " + epoch + "): loss: " + str(loss))
            lossSum = 0
            cycleCnt = 0

            merged_summary_op = tf.summary.merge_all()
            summary_merged = sess.run(merged_summary_op)
            writer.add_summary(summary_merged, int(i / 50))

        if i % 200 == 0:
            # average = lossSum / cycleCnt
            if not os.path.exists("../model/"+ model_dir):
                os.mkdir("../model/"+ model_dir)

            print("Saving checkpoint " + str(i))
            saver.save(sess, "../model/"+ model_dir + "model_" + str(i), write_meta_graph=False)


