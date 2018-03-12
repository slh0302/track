import tensorflow as tf
from model.RFCN import RFCN
from utils.CheckpointLoader import loadCheckpoint
from model.InceptionRFCN import InceptionRFCN
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

# code write by su
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

palette = Palette(len(categories))

# begin testing
image = tf.placeholder(tf.float32, [None, None, None, 3])
net = InceptionRFCN(image, len(categories), name="boxnet")
boxes, scores, classes = net.getBoxes(scoreThreshold=0.5)


with tf.Session() as sess:
    if not loadCheckpoint(sess, None, "../../save/model_3/model", ignoreVarsInFileNotInSess=True):
        print("Failed to load network.")
        sys.exit(-1)

    while True:
        img = cv2.imread("./img00122.jpg")

        if img is None:
            break

        img = preprocessInput(img)

        rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: np.expand_dims(img, 0)})

        res = drawBoxes(img, rBoxes, rClasses, [categories[i] for i in rClasses.tolist()], palette, scores=rScores)

        cv2.imwrite("./2.jpg", res)

        break