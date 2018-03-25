from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.MultiLoader.MultiBoxLoader import MultiBoxLoader
from MultiModel.MultiLoader.MultiTrack import MultiTrack
from datasets.process import Augment
from Visualize.Visualize import drawBoxes, Palette
import cv2
import numpy as np


def paddingDispSame(data):
    pad = [0.0, 0.0, 0.0, 0.0]
    tmp = [len(i) for i in data]
    maxLen = max(tmp)
    tmpList = []
    for index in range(len(data)):
        tmpList.append(data[index] + [pad] * (maxLen - tmp[index]))
    return tmpList, tmp


def paddingSame(data, pad=[0.0], length=None):
    if not length:
        tmp = [len(i) for i in data]
        maxLen = max(tmp)
        tmpList = []
        for index in range(len(data)):
            tmpList.append(data[index] + [pad] * (maxLen - tmp[index]))
        return tmpList, tmp
    else:
        maxLen = max(length)
        tmpList = []
        for index in range(len(data)):
            tmpList.append(data[index] + pad * (maxLen - length[index]))
        return tmpList, [0]

palette = Palette(4)
categories = ["others", "car", "van", "bus"]

mcar = MultiTrack("/home/slh/dataset/DETRAC/", set="Train", batch=4)
mcar.init()
while True:
    img, boxes, classes, ref1, ref2= mcar.load()
    boxes, number = paddingSame(boxes, pad=[0.0, 0.0, 0.0, 0.0])
    classes, _ = paddingSame(classes, pad=[100], length=number)
    ref1, num = paddingSame(ref1, pad=[0.0, 0.0, 0.0, 0.0])
    ref2, _ = paddingSame(ref2, pad=[[0.0, 0.0, 0.0, 0.0]], length=num)
    # images, boxes, classes, number = Augment.MutliAugment(img, boxes, classes, number)

    for im, rB, rC, rS, i in zip(img, boxes, classes, [1,1], range(2)):
        rB, rC =  rB[:number[i]], rC[:number[i]]
        res = drawBoxes(im, np.array(rB, dtype=np.float32), rC, [categories[i] for i in rC], palette)
        cv2.imwrite("./model_test" + str(i) + ".jpg", res)

    print("dasdasd")



    # <class 'list'>: ['MVI_20035/img00536.jpg', 'MVI_20035/img00537.jpg']
    # <class 'list'>: ['MVI_20035_img00536', 'MVI_20035_img00537'] 19 37675
    # <class 'list'>: ['MVI_40871/img01596.jpg', 'MVI_40871/img01597.jpg']
    # <class 'list'>: ['MVI_40871_img01596', 'MVI_40871_img01597'] 15 16516
    # <class 'list'>: ['MVI_39851/img00085.jpg', 'MVI_39851/img00086.jpg']
    # <class 'list'>: ['MVI_39851_img00085', 'MVI_39851_img00086'] 3 52770

