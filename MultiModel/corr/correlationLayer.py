from Correlation.Correlation import correlation
import tensorflow as tf
from Roi.ROIPoolingWrapper import positionSensitiveRoiPooling
from utils import BoxUtils
from basic import MultiGather
from Loss.Loss import smooth_l1
import tensorflow.contrib.slim as slim

class CorrelationNet:
    POOL_SIZE = 3

    def __init__(self, frame1, frame2, boxRegFeature, batch=8, max_displacement=8, inputDownscale=None, offset=None, reuse=False):
        self.inputDownScale = inputDownscale
        self.offset = offset
        self.max_displacement = max_displacement
        self.posIouTheshold = 0.7
        self.batch = batch
        self.reuse = reuse
        assert self.batch % 2 == 0
        self.define(frame1, frame2, boxRegFeature)

    def define(self, frame1, frame2, boxRegFeature):
        # frame1: 1,3,5,7
        # frame2: 2,4,6,8

        self.corrOut1 = correlation(frame1[0], frame2[0], kernel_size=1, max_displacement=self.max_displacement,
                              stride_1=1, stride_2=1, padding=8)
        self.corrOut2 = correlation(frame1[1], frame2[1], kernel_size=1, max_displacement=self.max_displacement,
                              stride_1=1, stride_2=1, padding=8)
        self.corrOut3 = correlation(frame1[2], frame2[2], kernel_size=3, max_displacement=self.max_displacement*2,
                              stride_1=2, stride_2=2, padding=16)

        # corrOut3.set_shape(tf.shape(corrOut2))

        corr_features = tf.concat([self.corrOut1, self.corrOut2, self.corrOut3], -1)
        corr_features = tf.concat([corr_features] + boxRegFeature, axis=-1)
        with tf.name_scope('RFCN_Disp'):
            # box prediction layers
            corr_feature = slim.conv2d(corr_features, 4 * (self.POOL_SIZE**2), 1, activation_fn=None, padding='SAME', scope='dispMaps',reuse=self.reuse)
            self.splitFeature = tf.split(corr_feature, int(self.batch/2), axis=0)

    def roiPooling(self, layer, boxes):
        return positionSensitiveRoiPooling(layer, boxes, offset=self.offset, downsample=self.inputDownScale,
                                           roiSize=self.POOL_SIZE)

    def roiMean(self, layer, boxes):
        with tf.name_scope("roiMean"):
            return tf.reduce_mean(self.roiPooling(layer, boxes), axis=[1, 2])

    def Loss(self, refDisp1, refDis2, proposals, DispNumber):
        # new
        tmp = []
        totalLos = []
        for item in proposals:
            tmp.append(tf.stop_gradient(item))
        proposals = tmp

        with tf.name_scope("trackingLoss"):
            refdisp1 = tf.unstack(refDisp1, axis=0)
            refdisp2 = tf.unstack(refDis2, axis=0)
            for disp1, disp2, pros, nm in zip(refdisp1, refdisp2, proposals, range(int(self.batch/2))):
                tm = tf.range(DispNumber[nm], dtype=tf.int32)
                disp1 = tf.gather(disp1, tm)
                disp2 = tf.gather(disp2, tm)

                iou = BoxUtils.iou(pros, disp1)
                maxIou = tf.reduce_max(iou, axis=1)
                bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

                # pos index
                posBoxIndices = tf.cast(tf.where(maxIou > self.posIouTheshold), tf.int32)
                posBoxes, posRefIndices = MultiGather.gather([pros, bestIou], posBoxIndices)
                posBoxes = tf.concat([posBoxes, disp1], 0)
                posRefIndices = tf.concat([posRefIndices, tf.reshape(tf.range(tf.shape(disp1)[0]), [-1, 1])],0)
                positiveRefIndices = tf.reshape(posRefIndices, [-1, 1])

                positives = self.roiMean(self.splitFeature[nm], posBoxes)
                # bbox
                frame1_box, frame2_box = MultiGather.gather([disp1, disp2], positiveRefIndices)
                los = self.computeLoss(positives, [frame1_box, frame2_box])
                totalLos.append(tf.expand_dims(los, 0))

            return tf.reduce_mean(tf.concat(totalLos, 0))
            # tf.constant(0.0)

    def computeLoss(self, box, refbox):
        x, y, w, h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(box, axis=1))
        bbox1 = refbox
        box1_x0, box1_y0, box1_w, box1_h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(bbox1[0], axis=1))
        box2_x0, box2_y0, box2_w, box2_h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(bbox1[1], axis=1))
        delta_x = (box2_x0 - box1_x0) / box1_w
        delta_y = (box2_y0 - box1_y0) / box1_h
        delta_w = tf.log(box2_w / box1_w)
        delta_h = tf.log(box2_h / box1_h)
        loss = smooth_l1(x - delta_x) + smooth_l1(y - delta_y) + smooth_l1(w - delta_w) + smooth_l1(h - delta_h)
        return tf.reduce_mean(loss)


    def getDisplacement(self, proposals):
        # proposals is first out in BoxregNet
        Disp = []
        proposals = tf.unstack(proposals, axis=0)
        for pros, nm in zip(proposals, range(int(self.batch/2))):
            positives = self.roiMean(self.splitFeature[nm], pros)
            Disp.append(positives)
        # fetch new data
        return Disp