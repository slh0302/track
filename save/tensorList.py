import tensorflow as tf
from model.TrackRPN import TrackRPN
from model.RFCN import RFCN
from utils.CheckpointLoader import loadCheckpoint
from model.InceptionRFCN import InceptionRFCN
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
inputs = np.array([128 for i in range(0, 540000)])
inputs = inputs.reshape([2,300,300,3]).astype(np.float32)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=8)) as sess:
    # MS COCO
    # basenet = InceptionRFCN(inputs, 80, hardMining=False, trainFrom=0)
    # if not loadCheckpoint(sess, "./model_3/", "./model_3/model"):
    #     print("Loading GoogleNet")
    #     basenet.importWeights(sess, "./model_1/inception_resnet_v2_2016_08_30.ckpt")
    #     # net.importWeights(sess, "initialWeights/", permutateRgb=False)
    #     print("Done.")

    # normal version
    basenet = InceptionRFCN(inputs, 4, base_model=TrackRPN, hardMining=False, trainFrom=0)
    # run global
    sess.run(tf.global_variables_initializer())
    basenet.importWeights(sess, "./model_1/inception_resnet_v2_2016_08_30.ckpt")


    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k,v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        # print(v)