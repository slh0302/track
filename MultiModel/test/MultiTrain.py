# loader
from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.MultiLoader.MultiBoxLoader import MultiBoxLoader
from MultiModel.MultiLoader.MultiCar import MultiCarDataset
from datasets.process import Augment
from tensorflow.python.ops import control_flow_ops
from utils.CheckpointLoader import loadCheckpoint
import tensorflow as tf
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "11,12"
MOVING_AVERAGE_DECAY = 0.99
num_gpus=2
gradClip=1
# def tower_loss(scope, images, labels):
#     """Calculate the total loss on a single tower running the CIFAR model.
#
#     Args:
#     scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
#     images: Images. 4D tensor of shape [batch_size, height, width, 3].
#     labels: Labels. 1D tensor of shape [batch_size].
#
#     Returns:
#      Tensor of shape [] containing the total loss for a batch of data
#     """
#
#     # Build inference Graph.
#     logits = cifar10.inference(images)
#
#     # Build the portion of the Graph calculating the losses. Note that we will
#     # assemble the total_loss using a custom function below.
#     _ = cifar10.loss(logits, labels)
#
#     # Assemble all of the losses for the current tower only.
#     losses = tf.get_collection('losses', scope)
#
#     # Calculate the total loss for the current tower.
#     total_loss = tf.add_n(losses, name='total_loss')
#
#     # Attach a scalar summary to all individual losses and the total loss; do the
#     # same for the averaged version of the losses.
#     for l in losses + [total_loss]:
#         # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#         # session. This helps the clarity of presentation on tensorboard.
#         loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
#         tf.summary.scalar(loss_name, l)
#
#     return total_loss

# def createUpdateOp(net, gradClip=1):
#     with tf.name_scope("optimizer"):
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         totalLoss = tf.losses.get_total_loss()
#         grads = optimizer.compute_gradients(totalLoss, var_list=net.getVariables())
#         if gradClip is not None:
#             cGrads = []
#             for g, v in grads:
#                 if g is None:
#                     print("WARNING: no grad for variable " + v.op.name)
#                     continue
#                 cGrads.append((tf.clip_by_value(g, -float(gradClip), float(gradClip)), v))
#             grads = cGrads
#
#         update_ops.append(optimizer.apply_gradients(grads))
#         return control_flow_ops.with_dependencies([tf.group(*update_ops)], totalLoss, name='train_op')

# 计算每一个变量梯度的平均值。
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

# Create a variable to count the number of train() calls. This equals the
# number of batches processed * FLAGS.num_gpus.
learning_rate = 0.0001
batch = 4
globalStep = tf.Variable(0, name='globalStep', trainable=False)
#globalStepInc = tf.assign_add(globalStep, 1)
print("learning: ", learning_rate)
# initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate,
                                     global_step=globalStep,
                                     decay_steps=10000, decay_rate=0.9)

dataset = MultiBoxLoader(batch=batch)
dataset.add(MultiCarDataset("/home/slh/dataset/DETRAC/", set="Train", batch=batch))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
# trainOp = createUpdateOp(net)

# Calculate the gradients for each model tower.
tower_grads = []
tower_loss = []

net_reuse = False
with tf.variable_scope(tf.get_variable_scope()):
    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ("Track", i)) as scope:
              # Dequeues one batch for the GPU
              # images, boxes, classes, number = Augment.MutliAugment(*dataset.get())
              nas = tf.get_variable_scope()
              images, boxes, classes, number = Augment.MutliAugment(*dataset.get())
              # TODO ?????? init?
              net = MultiBoxInceptionResnet(images, number, 4, name="boxnet", hardMining=False, batch=batch, reuse=net_reuse)
              net_reuse = True
              tf.get_variable_scope().reuse_variables()

              # scope loss fetch and produce
              loss = net.getLoss(boxes, classes)
              reg_loss = tf.losses.get_regularization_losses(scope)
              total_loss = loss + reg_loss
              tf.losses.add_loss(total_loss)

              # Assemble all of the losses for the current tower only.
              losses = tf.get_collection('losses', scope)

              # Calculate the total loss for the current tower.
              total_loss = tf.add_n(losses, name='total_loss')

              # tf.get_variable_scope().reuse_variables()
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
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_ops.append(optimizer.apply_gradients(grads,global_step=globalStep))
# update_ops.append(variables_averages_op)
train_op = control_flow_ops.with_dependencies([tf.group(*update_ops)], total_loss, name='train_op')
# Apply the gradients to adjust the shared variables.
# tf.summary.scalar("loss_t", train_op)
# merged_summary_op = tf.summary.merge_all()
# run phases
saver = tf.train.Saver(keep_checkpoint_every_n_hours=4, max_to_keep=10)
tf.summary.scalar("learning_1", learning_rate)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)) as sess:
    writer = tf.summary.FileWriter("../logs/", sess.graph)

    print("Loading GoogleNet")
    sess.run(tf.global_variables_initializer())

    #if not loadCheckpoint(sess, None, "../../save/model_3/model", ignoreVarsInFileNotInSess=True):
    net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
     #   print("reload network.")
      #  sys.exit(-1)

    # net.importWeights(sess, "initialWeights/", permutateRgb=False)
    print("Done.")
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
            i, ls, tlos = sess.run([globalStep, train_op, tower_loss])
        except KeyboardInterrupt:
            print("Keyboard interrupt. Shutting down.")
            sys.exit(0)

        lossSum += ls
        cycleCnt += 1
        print("train_op: ", ls)
        print("total_loss: ", tlos)
        print("GlobalStep: ", i)
        # print("Mu image: ", mimage)
        # average = 0
        if i % 10 == 0:
            if cycleCnt > 0:
                loss1 = lossSum / cycleCnt

            # lossS=sess.run(trainLossSum, feed_dict={
            # 	trainLossFeed: loss
            # })
            # log.add_summary(lossS, global_step=samplesSeen)

            epoch = "%.2f" % (float(i * batch * num_gpus * 2) / dataset.count())
            print("Iteration " + str(i) + " (epoch: " + epoch + "): loss: " + str(loss1))
            lossSum = 0
            cycleCnt = 0

        if i % 200 == 0:
            # average = lossSum / cycleCnt
            print("Saving checkpoint " + str(i))
            saver.save(sess, "../model/model_multi/model_" + str(i*batch*2), write_meta_graph=False)

        # tf.summary.scalar("loss_aver", average)
        merged_summary_op = tf.summary.merge_all()
        summary_merged = sess.run(merged_summary_op)
        writer.add_summary(summary_merged, i)


