import tensorflow as tf
from eval.eval import eval
from utils.CheckpointLoader import loadCheckpoint
from MultiModel.MultiBoxInceptionResnet import MultiBoxInceptionResnet
from MultiModel.corr.correlationLayer import CorrelationNet
import sys
import os
import cv2
import pickle
import numpy as np
import datetime
# np.split()
def iou(boxes, refBoxes, oneToAll=True):
        x0, y0, x1, y1 = boxes
        ref_x0, ref_y0, ref_x1, ref_y1 = np.split(refBoxes, 4, axis=-1)

        # Calculate box IOU
        x0 = np.reshape(x0, [-1, 1])
        y0 = np.reshape(y0, [-1, 1])
        x1 = np.reshape(x1, [-1, 1])
        y1 = np.reshape(y1, [-1, 1])

        if oneToAll:
            # row to cols change
            boxShape = [1, -1]
        else:
            boxShape = [-1, 1]

        ref_x0 = np.reshape(ref_x0, boxShape)
        ref_y0 = np.reshape(ref_y0, boxShape)
        ref_x1 = np.reshape(ref_x1, boxShape)
        ref_y1 = np.reshape(ref_y1, boxShape)

        # max_x0 has: shape(x0)[0] x shape(ref_x0)[0]
        max_x0 = np.maximum(x0, ref_x0)
        max_y0 = np.maximum(y0, ref_y0)
        min_x1 = np.minimum(x1, ref_x1)
        min_y1 = np.minimum(y1, ref_y1)

        intersect = np.maximum(min_x1 - max_x0 + 1, 0.0) * np.maximum(min_y1 - max_y0 + 1, 0.0)
        union = (x1 - x0 + 1) * (y1 - y0 + 1) + (ref_x1 - ref_x0 + 1) * (ref_y1 - ref_y0 + 1) - intersect

        iou = intersect / union

        return iou


def loadModImages(begin_img_id, anns, prefix=""):
    sp = begin_img_id.split('_')
    next_file = "img%05d" % (int(sp[-1][3:]) + 1)
    next_anns = ("%s_%s_%s" % (sp[0], sp[1], next_file))
    if not next_anns in anns:
        next_file = "img%05d" % (int(sp[-1][3:]) - 1)
        return begin_img_id, 0
    return begin_img_id, next_anns


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
batch = 1
TEST_SET = 'TestAns_120.pkl'
TEST_MODEL = '/home/slh/tf-project/track/MultiModel/model/track/reoldEnd2/model_2800'
TEST_NUMBER = 120
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
numClass = 1

if numClass ==4 :
    categories = ["__background__","others", "car", "van", "bus"]
elif numClass == 1:
    categories = ["__background__", "car"]
else:
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
print("class Number:", len(categories) -1 )
# begin testing
image = tf.placeholder(tf.float32, [None, None, None, 3])
net = MultiBoxInceptionResnet(image, len(categories) - 1, name="boxnet", batch=batch)
boxes, scores, classes = net.getBoxes(scoreThreshold=0.7)

""" corr """
frame1_1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1536], name="frame1_1")
frame1_2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1024], name="frame1_2")
frame1_3 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 320], name="frame1_3")

frame2_1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1536], name="frame2_1")
frame2_2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1024], name="frame2_2")
frame2_3 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 320], name="frame2_3")

# frame1_1 = tf.gather(net.featureInput, tf.range(batch, delta=2))
# frame1_2 = tf.gather(net.rpnInput, tf.range(batch, delta=2))
# frame1_3 = tf.gather(net.scale_32_2, tf.range(batch, delta=2))

# frame2_1 = tf.gather(net.featureInput, tf.range(1, batch, delta=2))
# frame2_2 = tf.gather(net.rpnInput, tf.range(1, batch, delta=2))
# frame2_3 = tf.gather(net.scale_32_2, tf.range(1, batch, delta=2))

frame1 = [frame1_1, frame1_2, frame1_3]
frame2 = [frame2_1, frame2_2, frame2_3]

boxMap1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, (3**2)*4], name="boxMap1")
boxMap2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, (3**2)*4], name="boxMap2")

# boxMap1 = tf.gather(net.boxRefiner.regressionMap, tf.range(batch, delta=2))
# boxMap2 = tf.gather(net.boxRefiner.regressionMap, tf.range(1, batch, delta=2))

dispBoxes = tf.placeholder(dtype=tf.float32, shape=[batch, None, 4], name="boxDisp")

corr = CorrelationNet(frame1, frame2, [boxMap1, boxMap2], batch=2 * batch, inputDownscale=16, offset=[32, 32])

Disp = corr.getDisplacement(dispBoxes)

""" end1 """


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
    if not loadCheckpoint(sess, None, TEST_MODEL, ignoreVarsInFileNotInSess=True):
        print("Failed to load network.")
        sys.exit(-1)

    # initGlobalVars()
    # net.importWeights(sess, "/home/slh/tf-project/track/save/model_1/inception_resnet_v2_2016_08_30.ckpt")
    first, second = loadModImages(imageName[begin], imageName)
    img1 = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + first + ".jpg")
    img1, zoom1, padTop1, padLeft1 = preprocessInput(img1)
    img1 = np.expand_dims(img1,axis=0)
    print("shape: ", img1.shape)
    rBoxes, rScores, rClasses, fin, rin, scale, reges = sess.run([boxes, scores, classes,
                                                    net.featureInput, net.rpnInput, net.scale_32_2,
                                                                     net.boxRefiner.regressionMap], feed_dict={image: img1})
    print("boxes: ", rBoxes[0].shape)
    # draw boxes
    for bx, tsc, cls in zip(rBoxes[0], rScores[0], rClasses[0]):
        # """ out put """
        x1, y1 = clipCoord(bx[0:2], img1[0])
        x2, y2 = clipCoord(bx[2:4], img1[0])
        cv2.rectangle(img1[0], tuple([x1, y1]), tuple([x2, y2]), (255, 255, 0), thickness=4)
    cv2.imwrite("/home/slh/tf-project/track/MultiModel/test/imgs/" + first + ".jpg", img1[0])
    del img1

    begin += 1
    while begin < runTestNum:
        tmp1, tmp2 = loadModImages(imageName[begin], imageName)
        if tmp1 == second:
            if tmp2 != 0:
                second = tmp2
            else:
                second = "0"
            img2 = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + tmp1 + ".jpg")
            img2, _, _, _ = preprocessInput(img2)
            img2 = np.expand_dims(img2, axis=0)

            rBoxes1, rScores1, rClasses1, fin1, rin1, scale1, reges1 = sess.run([boxes, scores, classes,
                                                                             net.featureInput, net.rpnInput,
                                                                             net.scale_32_2, net.boxRefiner.regressionMap], feed_dict={image: img2})

            print("feature: ", fin1.shape, rBoxes[0].shape)
            """ corr calu """
            disp = sess.run(Disp, feed_dict={frame1_1: fin, frame1_2: rin, frame1_3: scale,
                                             frame2_1: fin1, frame2_2: rin1, frame2_3: scale1,
                                             boxMap1: reges, boxMap2: reges1, dispBoxes: rBoxes})
            print("feature: ", [i.shape for i in disp])
            # draw boxes
            for bx, sc, cls in zip(rBoxes1[0], rScores1[0], rClasses1[0]):
                sc = "%.2f" % sc
                # """ out put """
                x1, y1 = clipCoord(bx[0:2], img2[0])
                x2, y2 = clipCoord(bx[2:4], img2[0])
                cv2.rectangle(img2[0], tuple([x1, y1]), tuple([x2, y2]), (255, 255, 0), thickness=4)

            x1, y1, x2, y2 = np.split(rBoxes[0],4, axis=-1)
            w1, h1 = x2 - x1, y2 - y1
            delt_x, delt_y, delt_w, delt_h = disp[0]
            x = x1 + delt_x * w1
            y = y1 + delt_y * h1

            w = w1 * tf.exp(delt_w)
            h = h1 * tf.exp(delt_h)

            bxiou = iou([x,y,w,h], rBoxes1[0])
            maxIou = np.max(bxiou, axis=1)
            bestIou = np.cast(np.argmax(bxiou, axis=1), np.int32)
            posBoxIndices = np.cast(np.where(maxIou > 0.4), np.int32)
            for index in posBoxIndices:
                nx = x[index]
                ny = y[index]
                nx2 = nx + w[index]
                ny2 = ny + h[index]
                cv2.rectangle(img2[0], tuple([nx, ny]), tuple([nx2, ny2]), (0, 255, 255), thickness=4)

            cv2.imwrite("/home/slh/tf-project/track/MultiModel/test/imgs/" + tmp1 + ".jpg", img2[0])
            rBoxes, rScores, rClasses, fin, rin, scale, reges = rBoxes1, rScores1, rClasses1, fin1, rin1, scale1, reges1
            del img2
        else:
            first, second = loadModImages(imageName[begin], imageName)
            img1 = cv2.imread("/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + first + ".jpg")
            img1, zoom1, padTop1, padLeft1 = preprocessInput(img1)
            img1 = np.expand_dims(img1, axis=0)
            rBoxes, rScores, rClasses, fin, rin, scale, reges = sess.run([boxes, scores, classes,
                                                                                    net.featureInput, net.rpnInput,
                                                                                    net.scale_32_2,
                                                                                    net.boxRefiner.regressionMap],
                                                                                   feed_dict={image: img1})
            # draw boxes
            for bx, tsc, cls in zip(rBoxes[0], rScores[0], rClasses[0]):
                # """ out put """
                x1, y1 = clipCoord(bx[0:2], img1[0])
                x2, y2 = clipCoord(bx[2:4], img1[0])
                cv2.rectangle(img1[0], tuple([x1, y1]), tuple([x2, y2]), (255, 255, 0), thickness=4)
            cv2.imwrite("/home/slh/tf-project/track/MultiModel/test/imgs/" + first + ".jpg", img1[0])
            del img1

        begin += 1

sess.close()

# print("total_run: ", begin)
# eva = eval(categorieOut, "/home/slh/tf-project/track/MultiModel/TFRecord", "test")
# eva.evaluate_SingleDetections(boxesRes, '/home/slh/tf-project/track/MultiModel/test/output', ansname=TEST_SET)
#
endTime = datetime.datetime.now()
k = endTime - beginTime
print("Time: ", k)