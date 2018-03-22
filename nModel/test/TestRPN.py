import tensorflow as tf
from model.RFCN import RFCN
from utils.CheckpointLoader import loadCheckpoint
from nModel.BoxInceptionResnet import BoxInceptionResnet
from Visualize.Visualize import drawBoxes, Palette
import sys
import os
import cv2
import numpy as np

def preprocessInput(img):
    def calcPad(size):
        m = size % 32
        p = int(m/2)
        s = size - m
        return s,p

    zoom = max(640.0 / img.shape[0], 640.0 / img.shape[1])
    img = cv2.resize(img, (int(zoom*img.shape[1]), int(zoom*img.shape[0])))

    if img.shape[0] % 32 != 0:
        s,p = calcPad(img.shape[0])
        img = img[p:p+s]

    if img.shape[1] % 32 != 0:
        s,p = calcPad(img.shape[1])
        img = img[:,p:p+s]

    return img


def initGlobalVars():
    if "global_variables_initializer" in tf.__dict__:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.initialize_all_variables())

# code write by su
os.environ["CUDA_VISIBLE_DEVICES"] = "12"
categories = ["others", "car", "van", "bus"]
# categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
#  'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
#  'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
#  'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

palette = Palette(len(categories))

# begin testing
image = tf.placeholder(tf.float32, [None, None, None, 3])
net = BoxInceptionResnet(image, len(categories), name="boxnet")
boxes, scores, classes = net.getBoxes(scoreThreshold=0.5)


with tf.Session() as sess:
    if not loadCheckpoint(sess, None, "../model/model_binary/model_12000", ignoreVarsInFileNotInSess=True):
        print("Failed to load network.")
        sys.exit(-1)

    # initGlobalVars()
    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")

    while True:
        img = cv2.imread("./img00022.jpg")

        if img is None:
            break

        img = preprocessInput(img)

        rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: np.expand_dims(img, 0)})

        res = drawBoxes(img, rBoxes, rClasses, [categories[i] for i in rClasses.tolist()], palette, scores=rScores)

        cv2.imwrite("./model_binary.jpg", res)

        break