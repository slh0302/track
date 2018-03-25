from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.MultiLoader.MultiBoxLoader import MultiBoxLoader
from MultiModel.MultiLoader.MultiCar import MultiCarDataset
from datasets.process import Augment
from tensorflow.python.ops import control_flow_ops
from utils.CheckpointLoader import loadCheckpoint
import tensorflow as tf
import sys
import os

# param
log_dir = "mult_4/"
model_dir = "model_mulit_4/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,8,9"
MOVING_AVERAGE_DECAY = 0.99
num_gpus=3
gradClip=1
NumClasses=4
learning_rate = 0.01
batch = 2
opts="sgd"

def average_gradients(tower_grads):
    average_grads = []

    # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
    for grad_and_vars in zip(*tower_grads):
        # 计算所有GPU上的梯度平均值。
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # 将变量和它的平均梯度对应起来。
        average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这个将被用于变量的更新。
    return average_grads


"""Train Multi car"""
with tf.Graph().as_default() as graph:
    globalStep = tf.Variable(0, name='globalStep', trainable=False)
    # globalStepInc = tf.assign_add(globalStep, 1)
    print("learning: ", learning_rate)

    # loader
    dataset = MultiBoxLoader(batch=batch)
    dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch, Noc=NumClasses))

    if opts == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # initial_learning_rate = 0.
    num_epoch_for_decay = 1
    dataset.init()
    decay_steps = int((dataset.count() / (batch * 2 * num_gpus)) * num_epoch_for_decay)
    learning_rate = tf.train.exponential_decay(learning_rate,
                                         global_step=globalStep,
                                         decay_steps=decay_steps, decay_rate=0.8, staircase=True)


    # Calculate the gradients for each model tower.
    tower_grads = []
    tower_loss = []
    net_reuse = False
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("Track", i)) as scope:
                  # Dequeues one batch for the GPU
                  images, boxes, classes, number = Augment.MutliAugment(*dataset.get())

                  net = MultiBoxInceptionResnet(images, NumClasses, name="boxnet", hardMining=True,
                                                batch=batch, reuse=net_reuse, trainFrom=0)
                  net_reuse = True
                  tf.get_variable_scope().reuse_variables()

                  # scope loss fetch and produce
                  loss = net.getLoss(boxes, classes, number)
                  tf.losses.add_loss(loss)
                  reg_loss = tf.losses.get_regularization_losses(scope)
                  total_loss = tf.losses.get_losses() + reg_loss

                  #TODO: NOT ADD IS THE TOTAL LOSS
                  tf.losses.add_loss(total_loss)

                  # Assemble all of the losses for the current tower only.
                  losses = tf.get_collection('losses', scope)

                  # Calculate the total loss for the current tower.
                  total_loss = tf.add_n(losses, name='total_loss')

                  grads = optimizer.compute_gradients(total_loss, var_list=net.getVariables())
                  if gradClip is not None:
                      cGrads = []
                      for g, v in grads:
                          if g is None:
                              print("WARNING: no grad for variable " + v.op.name)
                              continue
                          cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
                      grads = cGrads

                  # Keep track of the gradients across all towers.
                  tower_grads.append(grads)
                  tower_loss.append(total_loss)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    # 计算变量的滑动平均值。
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, globalStep)
    # variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
    # variables_averages_op = variable_averages.apply(variables_to_average)
    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #       summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # update_ops.append(variables_averages_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(optimizer.apply_gradients(grads, global_step=globalStep))

    res_loss = tf.reduce_mean(tower_loss)
    train_op = control_flow_ops.with_dependencies([tf.group(*update_ops)], res_loss, name='train_op')


    tf.summary.scalar("loss_t", res_loss)
    tf.summary.scalar("learning_1", learning_rate)

# run phases


config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=graph) as sess:
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=10)
    if not os.path.exists("../logs/" + log_dir):
        os.mkdir("../logs/" + log_dir)
    writer = tf.summary.FileWriter("../logs/" + log_dir, sess.graph)

    # init net
    print("Loading GoogleNet")
    sess.run(tf.global_variables_initializer())

    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
    if not loadCheckpoint(sess, None, "../../save/model_3/model", ignoreVarsInFileNotInSess=True, ignoreClass=True):
       print("reload network.")
       sys.exit(-1)
    # net.importWeights(sess, "initialWeights/", permutateRgb=False)
    print("Done.")

    i = sess.run(globalStep)
    merged_summary_op = tf.summary.merge_all()
    summary_merged = sess.run(merged_summary_op)
    writer.add_summary(summary_merged, i)
    sess.graph.finalize()

    dataset.startThreads(sess)
    cycleCnt = 0
    lossSum = 0

    while True:
        try:
            i, ls, tbox, nm, rpnls, bxls = sess.run(
                [globalStep, train_op, net.totalBox, net.testnumber, net.rpnloss, net.boxLoss])

        except KeyboardInterrupt:
            sess.close()
            del sess
            print("Keyboard interrupt. Shutting down.")
            sys.exit(0)

        print("setp ", i, " loss: ", ls, " testnumber: ", nm.tolist(), " toal box:",
              [tbox[i].shape for i in range(batch)],
              " rpn loss:", rpnls, " boxls:", bxls)

        lossSum += ls
        cycleCnt += 1
        if i % 10 == 0:
            print("tesing")
            if cycleCnt > 0:
                loss1 = lossSum / cycleCnt

            epoch = "%.2f" % (float(i * batch * num_gpus * 2) / dataset.count())
            print("Iteration " + str(i) + " (epoch: " + epoch + "): loss: " + str(loss1))
            lossSum = 0
            cycleCnt = 0

        if i % 200 == 0:
            if not os.path.exists("../model/"+ model_dir):
                os.mkdir("../model/"+ model_dir)
            print("Saving checkpoint " + str(i))
            saver.save(sess, "../model/"+ model_dir + "model_" + str(i), write_meta_graph=False)




