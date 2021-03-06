from utils.Resnetv2 import *
from utils.InceptionResnetV2 import *
from model.RPN import RPN
import tensorflow.contrib.slim as slim
import tensorflow as tf


class InceptionTrackRFCN:
    LAYER_NAMES = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 'MaxPool_3a_3x3', 'Conv2d_3b_1x1',
                   'Conv2d_4a_3x3',
                   'MaxPool_5a_3x3', 'Mixed_5b', 'Repeat', 'Mixed_6a', 'Repeat_1', 'Mixed_7a', 'Repeat_2', 'Block8',
                   'Conv2d_7b_1x1']

    def __init__(self, inputs, nCategories, name="boxnet", weightDecay=0.00004,
                 reuse=False, isTraining=True, trainFrom=None, hardMining=True, freezeBatchNorm=False):
        self.boxThreshold = 0.5

        try:
            trainFrom = int(trainFrom)
        except:
            pass

        if isinstance(trainFrom, int):
            trainFrom = self.LAYER_NAMES[trainFrom]

        print("Training network from " + (trainFrom if trainFrom is not None else "end"))

        # print("Training network from " + (trainFrom if trainFrom is not None else "end"))
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.Inception_model = InceptionResnetV2("features", inputs, trainFrom=trainFrom, freezeBatchNorm=freezeBatchNorm)
            self.scope = scope

            with tf.variable_scope("Box"):
                # Pepeat_1 - last 1/16 layer, Mixed_6a - first 1/16 layer
                scale_16 = self.Inception_model.getOutput("Repeat_1")[:,1:-1,1:-1,:]
                scale_32 = self.Inception_model.getOutput("PrePool")
                # bn + relu
                print(scale_16.shape, scale_32.shape)
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(weightDecay),
                                    biases_regularizer=slim.l2_regularizer(weightDecay),
                                    padding='SAME',
                                    activation_fn=tf.nn.relu):
                    net = tf.concat([tf.image.resize_bilinear(scale_32, tf.shape(scale_16)[1:3]), scale_16], 3)
                    rpnInput = slim.conv2d(net, 1024, 1)
                    featureInput = slim.conv2d(net, 1536, 1)
                    self.Rpn = RPN(nCategories, rpnInput, 16, [32, 32], featureInput, 16, [32, 32],
                                        weightDecay=weightDecay, hardMining=hardMining)

    def importWeights(self, sess, filename):
        self.Inception_model.importWeights(sess, filename, includeTraining=True)

    def getLoss(self,refBoxes, refClasses):
        return self.Rpn.getLoss(refBoxes, refClasses)

    def getProposals(self, threshold=None):
        if threshold is not None and threshold > 0:
            s = tf.cast(tf.where(self.Rpn.proposalScores > threshold), tf.int32)
            return tf.gather_nd(self.Rpn.proposals, s), tf.gather_nd(self.Rpn.proposalScores, s)
        else:
            return self.Rpn.proposals, self.Rpn.proposalScores

    def getBoxes(self, nmsThreshold=0.3, scoreThreshold=0.8):
        return self.Rpn.BoxRegNet.getBoxes(self.Rpn.proposals, self.Rpn.proposalScores, maxOutputs=50, nmsThreshold=nmsThreshold,
                                        scoreThreshold=scoreThreshold)
