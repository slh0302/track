import tensorflow as tf
from model.RFCN import RFCN
from utils.CheckpointLoader import loadCheckpoint
from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from Visualize.Visualize import drawBoxes, Palette
import sys
import os
import cv2
import numpy as np
import pickle

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


batch =2
# code write by su
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# categories = ["background", "others", "car", "van", "bus"]
categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

palette = Palette(len(categories))

# begin testing
image = tf.placeholder(tf.float32, [None, None, None, 3])
net = MultiBoxInceptionResnet(image, len(categories), name="boxnet", batch=batch)
boxes, scores, classes = net.getBoxes(scoreThreshold=0.6)



f = open('/home/slh/tf-project/track/MultiModel/TFRecord/TestAns.pkl', 'rb')
testSet = pickle.load(f)
f.close()

imageName = list(testSet.keys()) #.tolist()
total =  len(imageName)
begin = 0
runTestNum = total if total < 120 else 120
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session() as sess:
    if not loadCheckpoint(sess, None, "../model/model_old_hard_80/model_32000", ignoreVarsInFileNotInSess=True):
        print("Failed to load network.")
        sys.exit(-1)

    # initGlobalVars()
    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")

    while begin < runTestNum:
        imList = []
        resultList = []
        for i in range(batch):
            name = imageName[begin + i]
            resultList.append(name)
            img1 = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + name + ".jpg")
            if img1 is None:
                continue
            img = preprocessInput(img1)
            imList.append(img)
        begin += batch
        rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: imList})
        for im, rB, rC, rS, imname in zip(imList, rBoxes, rClasses, rScores, resultList):
            res = drawBoxes(im, rB, rC, [categories[i] for i in rC.tolist()], palette, scores=rS)
            cv2.imwrite("./imgs/"+ imname + ".jpg", res)

    # while True:
    #     img = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + imageName[100] + ".jpg")
    #     img2 = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + imageName[101] + ".jpg")
    #     if img is None:
    #         break
    #
    #     img = preprocessInput(img)
    #     img2 = preprocessInput(img2)
    #     picture = [img, img2]
    #
    #     rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: picture})
    #     for im, rB, rC, rS, i in zip(picture, rBoxes, rClasses, rScores, range(2)):
    #         res = drawBoxes(im, rB, rC, [categories[i] for i in rC.tolist()], palette, scores=rS)
    #         cv2.imwrite("./model_single_mod12111" + str(i) + ".jpg", res)
    #     break