import tensorflow as tf
from model.RFCN import RFCN
from model.TrackRPN import TrackRPN
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
os.environ["CUDA_VISIBLE_DEVICES"] = "14"

categories = ["null", "others", "car", "van", "bus"]

palette = Palette(len(categories))

# begin testing
image = tf.placeholder(tf.float32, [2, None, None, 3])
net = InceptionRFCN(image, 4, base_model=TrackRPN, hardMining=False, trainFrom=0)
boxes, scores, classes = net.getBoxes(scoreThreshold=0.5)


with tf.Session() as sess:
    if not loadCheckpoint(sess, None, "../../save/model_2000", ignoreVarsInFileNotInSess=True):
        print("Failed to load network.")
        sys.exit(-1)

    while True:
        img = cv2.imread("./img00022.jpg")
        img2 = cv2.imread("./img00023.jpg")

        if img is None:
            break

        img = preprocessInput(img)
        img2 = preprocessInput(img2)

        rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: [img, img2]})
        for imt, rb,rs,rc, num in zip([img, img2], rBoxes, rScores, rClasses, range(2)):
            res = drawBoxes(imt, rb, rc, [categories[i] for i in rc.tolist()], palette, scores=rs)

            cv2.imwrite("./" + str(num) + ".jpg", res)

        break