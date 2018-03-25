from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.MultiLoader.MutliTrackLoader import MultiTrackLoader
from MultiModel.MultiLoader.MultiTrack import MultiTrack
from datasets.process import Augment
from basic.MultiGather import gather
from tensorflow.python.ops import control_flow_ops
from utils.CheckpointLoader import loadCheckpoint
from MultiModel.corr.correlationLayer import CorrelationNet
import tensorflow as tf
import sys
import os

log_dir = "track/reoldEnd2/"
model_dir = "track/reoldEnd2/"
NumClasses = 1
trainFrom=None
learning_rate = 0.0001
batch = 8
LoadOld=True
ignoreClass=True
staircase=True
opts="sgd"
modelToLoad = "/home/slh/tf-project/track/save/model_3/model"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# def test_Block():
#
# graph
with tf.Graph().as_default() as graph:
    globalStep = tf.Variable(0, name='globalStep', trainable=False)
    # globalStepInc = tf.assign_add(globalStep, 1)
    print("learning: ", learning_rate)
    # initial_learning_rate = 0.1
    # load data
    dataset = MultiTrackLoader(batch=batch)
    dataset.add(MultiTrack("/home/slh/dataset/DETRAC/", set="Train", batch=batch, Noc=NumClasses))
    # dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch))
    image, boxes, classes, number, refDisp1, refDisp2, refDispNumber = dataset.get()
    images, boxes, classes, number = Augment.MutliAugment(image, boxes, classes, number)

    num_epoch_for_decay = 2
    # call to calc
    dataset.init()
    decay_steps = int((dataset.count() / (batch * 2 * 1)) * num_epoch_for_decay)
    learning_rate = tf.train.exponential_decay(learning_rate,
                                               global_step=globalStep,
                                               decay_steps=decay_steps, decay_rate=0.8, staircase=True)

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
    
    # track Corr
    frame1_1 = tf.gather(net.featureInput, tf.range(batch, delta=2))
    frame1_2 = tf.gather(net.rpnInput, tf.range(batch, delta=2))
    frame1_3 = tf.gather(net.scale_32_2, tf.range(batch, delta=2))

    frame2_1 = tf.gather(net.featureInput, tf.range(1, batch, delta=2))
    frame2_2 = tf.gather(net.rpnInput, tf.range(1, batch, delta=2))
    frame2_3 = tf.gather(net.scale_32_2, tf.range(1, batch, delta=2))

    frame1 = [frame1_1, frame1_2, frame1_3]
    frame2 = [frame2_1, frame2_2, frame2_3]

    boxMap1 = tf.gather(net.boxRefiner.regressionMap, tf.range(batch, delta=2))
    boxMap2 = tf.gather(net.boxRefiner.regressionMap, tf.range(1, batch, delta=2))


    corr = CorrelationNet(frame1, frame2, [boxMap1, boxMap2], batch=batch, inputDownscale=16, offset=[32,32])
    loss, bxes = net.getLoss(boxes, classes, number)
    for item in bxes:
        item.set_shape([None, 4])
    tbxes = [bxes[i] for i in range(0, batch, 2)]
    loss_corr = corr.Loss(refDisp1, refDisp2, tbxes, refDispNumber)
    #+ loss_corr
    tf.losses.add_loss(loss + loss_corr)
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
    if not os.path.exists("../../logs/" + log_dir):
        os.mkdir("../../logs/" + log_dir)
    writer = tf.summary.FileWriter("../../logs/" + log_dir, sess.graph)

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
    shape21, shape22, shape34, shape56 = sess.run([tf.shape(corr.corrOut1),tf.shape(corr.corrOut3), tf.shape(net.scale_32_2), tf.shape(net.rpnInput)])
    #
    # hIn, wIn = sess.run([net.rpn.hIn, net.rpn.wIn])
    # tf.summary.scalar("loss_aver", average)
    # make graph read only

    i = sess.run(globalStep)

    # sess.graph.finalize()

    cycleCnt = 0
    lossSum = 0
    while True:
        try:
            i, ls, cor_lss= sess.run([globalStep, trainOp, loss_corr])
        except KeyboardInterrupt:
            sess.close()
            writer.close()
            del sess
            print("Keyboard interrupt. Shutting down")
            sys.exit(0)

        lossSum += ls
        cycleCnt += 1
        # average = 0
        print("setp ",i," loss: ", ls, " cor_lss: ", cor_lss)

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
            if not os.path.exists("../../model/"+ model_dir):
                os.mkdir("../../model/"+ model_dir)

            print("Saving checkpoint " + str(i))
            saver.save(sess, "../../model/"+ model_dir + "model_" + str(i), write_meta_graph=False)