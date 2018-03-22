import tensorflow as tf
from eval.eval import eval
from utils.CheckpointLoader import loadCheckpoint
from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
import sys
import os
import cv2
import pickle
import numpy as np
import datetime


def preprocessInput(img):
    def calcPad(size):
        m = size % 32
        p = int(m / 2)
        s = size - m
        return s, p

    zoom = max(640.0 / img.shape[0], 640.0 / img.shape[1])
    img = cv2.resize(img, (int(zoom * img.shape[1]), int(zoom * img.shape[0])))

    padTop = 0
    padLeft = 0
    if img.shape[0] % 32 != 0:
        s, p = calcPad(img.shape[0])
        img = img[p:p + s]
        padTop = p

    if img.shape[1] % 32 != 0:
        s, p = calcPad(img.shape[1])
        img = img[:, p:p + s]
        padLeft = p

    return img, zoom, padTop, padLeft


def clipCoord(xy, img):
    return np.minimum(np.maximum(np.array(xy, dtype=np.int32), 0), [img.shape[1] - 1, img.shape[0] - 1]).tolist()


# originSize height * width
def originCoord(x, y, x1, y1, zoom, padTop, padLeft, originSize):
    box = [(x + padLeft) / zoom, (y + padTop) / zoom, (x1 + padLeft) / zoom, (y1 + padTop) / zoom]
    box[0] = max(min(box[0], originSize[1]), 0)
    box[1] = max(min(box[1], originSize[0]), 0)
    box[2] = max(min(box[2], originSize[1]), 0)
    box[3] = max(min(box[3], originSize[0]), 0)
    return box


def initGlobalVars():
    if "global_variables_initializer" in tf.__dict__:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.initialize_all_variables())


# code write by su
batch = 8
TEST_SET = 'TestAns.pkl'
TEST_NUMBER = 500000
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# categories = ["__background__","others", "car", "van", "bus"]
categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
              'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
              'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair',
              'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster',
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

beginTime = datetime.datetime.now()

# begin testing
image = tf.placeholder(tf.float32, [None, None, None, 3])
net = MultiBoxInceptionResnet(image, len(categories) - 1, name="boxnet", batch=batch)
boxes, scores, classes = net.getBoxes(scoreThreshold=0.6)
categorieOut = ["__background__", "car"]
boxesRes = {"__background__": {}}
for Class in categorieOut:
    boxesRes[Class] = {}

f = open('/home/slh/tf-project/track/MultiModel/TFRecord/' + TEST_SET, 'rb')
testSet = pickle.load(f)
f.close()

imageName = list(testSet.keys())  # .tolist()
begin = 0
total = len(imageName)
runTestNum = total if total < TEST_NUMBER else TEST_NUMBER

# font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontSize = 0.8
fontThickness = 1
pad = 5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if not loadCheckpoint(sess, None, "../model/model_old_hard_80/model_32000", ignoreVarsInFileNotInSess=True):
        print("Failed to load network.")
        sys.exit(-1)

    # initGlobalVars()
    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
    while begin + batch <= runTestNum:
        imList = []
        resultList = []
        param = []
        sizeList = []
        for i in range(batch):
            name = imageName[begin + i]
            resultList.append(name)
            img2 = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + name + ".jpg")
            sizeList.append(img2.shape)
            if img2 is None:
                continue

            img2, zoom, padTop, padLeft = preprocessInput(img2)
            imList.append(img2)
            param.append([zoom, padTop, padLeft])

        begin += batch
        if begin % 1200 == 0:
            print("Test Done: ", begin)

        rBoxes, rScores, rClasses = sess.run([boxes, scores, classes], feed_dict={image: imList})
        for im, imName, rB, rC, rS, i, par, sList in zip(imList, resultList, rBoxes, rClasses, rScores, range(batch),
                                                         param, sizeList):
            item = {}
            for bx, sc, cls in zip(rB, rS, rC):
                cls += 1
                sc = "%.2f" % sc

                boxCoord = originCoord(bx[0], bx[1], bx[2], bx[3], par[0], par[1], par[2], sList)
                if imName in boxesRes['car'].keys():
                    boxesRes['car'][imName].append(boxCoord + [float(sc)])
                else:
                    boxesRes['car'][imName] = [boxCoord + [float(sc)]]

print("total_run: ", begin)
eva = eval(categorieOut, "/home/slh/tf-project/track/MultiModel/TFRecord", "test")
eva.evaluate_SingleDetections(boxesRes, '/home/slh/tf-project/track/MultiModel/test/output', ansname=TEST_SET)

endTime = datetime.datetime.now()
k = endTime - beginTime
print("Time: ", k)