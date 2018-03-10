from model.MultiBoxRegNet import MultiBoxRegNet
from model.Multi import MultiRPN
from Correlation.Correlation import *
from Roi.ROIPoolingWrapper import *
from Loss.Loss import boxRegressionLoss, smooth_l1
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import basic.RandomSelect
import utils.BoxUtils as BoxUtils
import Loss.Loss as Loss
import basic.MultiGather as MultiGather

#  writer: su

class TrackRPN:
    def __init__(self, nClass, rpnLayer, rpnDownscale, rpnOffset, featureLayer=None, featureDownsample=None,
                 featureOffset=None, weightDecay=1e-6, hardMining=True):

        # MultifeatureLayer: All resnet block results
        MultifeatureLayer = featureLayer
        if MultifeatureLayer is None:
            MultifeatureLayer = [rpnLayer]

        if featureDownsample is None:
            featureDownsample = rpnDownscale

        if featureOffset is None:
            rpnOffset = featureOffset

        with tf.name_scope("RPN_module"):
            self.rpn = MultiRPN(rpnLayer, immediateSize=512, weightDecay=weightDecay,
                               inputDownscale=rpnDownscale, offset=rpnOffset)
            self.BoxRegNet = MultiBoxRegNet(MultifeatureLayer[0], nClass, downsample=featureDownsample,
                                       offset=featureOffset, hardMining=hardMining)
            self.Corr = CorrelationNet(MultifeatureLayer[1:], self.BoxRegNet.regressionMap, max_displacement=8, weightDecay=weightDecay,
                               inputDownscale=rpnDownscale, offset=rpnOffset)
            # list type
            self.proposals, self.proposalScores = self.rpn.getPositiveOutputs(maxOutSize=300)


    # calculate proposals
    def getProposals(self, threshold=None):
        if threshold is not None and threshold > 0:
            s = tf.cast(tf.where(self.proposalScores > threshold), tf.int32)
            return tf.gather_nd(self.proposals, s), tf.gather_nd(self.proposalScores, s)
        else:
            return self.proposals, self.proposalScores

    def getBoxes(self, nmsThreshold=0.3, scoreThreshold=0.8):
        return self.BoxRegNet.getBoxes(self.proposals, self.proposalScores, maxOutputs=50, nmsThreshold=nmsThreshold,
                                       scoreThreshold=scoreThreshold)

    def getDisplacement(self, getedBox):
        return self.Corr.getDisplacement(getedBox)

    def getLoss(self, refBoxes, refClasses, refDisp):
        loss, box = self.BoxRegNet.loss(self.proposals, refBoxes, refClasses)
        loss_corr = self.Corr.Loss(refDisp, box[0])
        return self.rpn.loss(refBoxes) + loss + loss_corr


class CorrelationNet:
    POOL_SIZE = 3

    def __init__(self, multi_featureLayers, rfcn_bbox_frame, max_displacement=8, weightDecay=1e-5, inputDownscale=None, offset=None):
        self.features = multi_featureLayers
        self.inputDownScale = inputDownscale
        self.offset = offset
        self.max_displacement = max_displacement
        self.posIouTheshold = 0.5


        self.define(rfcn_bbox_frame, weightDecay)

    def define(self, rfcn_bbox, weightDecay):
        if len(self.features) < 3:
            print("features wrong")
            return

        # block order: 5 4 3
        block5_feature = tf.split(self.features[0],2,0)
        block4_feature = tf.split(self.features[1],2,0)
        block3_feature = tf.split(self.features[2],2,0)

        # rfcn bbox two frames
        rfcn_bbox_feature = tf.split(rfcn_bbox,2,0)
        num_output = rfcn_bbox.get_shape().as_list()[-1]

        # correlation
        output_5 = correlation(block5_feature[0], block5_feature[1],
                               kernel_size=1, max_displacement=self.max_displacement,
                               stride_1=1, stride_2=1, padding=8)
        output_4 = correlation(block4_feature[0], block4_feature[1],
                               kernel_size=1, max_displacement=self.max_displacement,
                               stride_1=1, stride_2=1, padding=8)
        output_3 = correlation(block3_feature[0], block3_feature[1],
                               kernel_size=1, max_displacement=self.max_displacement,
                               stride_1=1, stride_2=1, padding=8)

        corr_features = tf.concat([output_5, output_4, output_3], -1)
        corr_features = tf.concat([rfcn_bbox_feature[0], rfcn_bbox_feature[1], corr_features], axis=-1)
        with tf.name_scope('RFCN_Disp'):
            # box prediction layers
            self.corr_feature = slim.conv2d(corr_features, num_output, 1, activation_fn=None, padding='SAME')

    def roiPooling(self, layer, boxes):
        return positionSensitiveRoiPooling(layer, boxes, offset=self.offset, downsample=self.inputDownScale,
                                           roiSize=self.POOL_SIZE)

    def roiMean(self, layer, boxes):
        with tf.name_scope("roiMean"):
            return tf.reduce_mean(self.roiPooling(layer, boxes), axis=[1, 2])

    def Loss(self, refDisp, proposals):
        loss = 0.0
        # new
        proposals = tf.stop_gradient(proposals)
        with tf.name_scope("trackingLoss"):
            iou = BoxUtils.iou(proposals, refDisp[0])
            maxIou = tf.reduce_max(iou, axis=1)
            bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

            # pos index
            posBoxIndices = tf.cast(tf.where(maxIou > self.posIouTheshold), tf.int32)
            posBoxes, posRefIndices = MultiGather.gather([proposals, bestIou], posBoxIndices)
            posBoxes = tf.concat([posBoxes, refDisp[0]], 0)
            posRefIndices = tf.concat([posRefIndices, tf.reshape(tf.range(tf.shape(refDisp[0])[0]), [-1, 1])],0)
            positiveRefIndices = tf.reshape(posRefIndices, [-1, 1])

            positives = self.roiMean(self.corr_feature, posBoxes)
            self.posboxNum = []
            self.posboxNum.append(tf.shape(proposals))
            self.posboxNum.append(tf.shape(positives))
            # bbox
            frame1_box, frame2_box = MultiGather.gather([refDisp[0], refDisp[1]], positiveRefIndices)
            self.posboxNum.append(tf.shape(frame1_box))
            self.posboxNum.append(tf.shape(frame2_box))
            los = self.computeLoss(positives, [frame1_box, frame2_box])
            self.posboxNum.append(los)
            return tf.cond(tf.shape(positives)[0] > 0, lambda: self.computeLoss(positives, [frame1_box, frame2_box]), lambda: tf.constant(0.0))
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

    def getRefineBox(self, proposals, needSizes):
        with tf.name_scope("trackingBoxes"):
            proposals_bboxs = self.roiMean(self.corr_feature, proposals)
            # fetch new data
            x, y, w, h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(proposals, axis=1))
            x_rel, y_rel, w_rel, h_rel = tf.unstack(proposals_bboxs, axis=1)

            if needSizes:
                refSizes = tf.stack([h, w], axis=1)

            x = x + x_rel * w
            y = y + y_rel * h

            w = w * tf.exp(w_rel)
            h = h * tf.exp(h_rel)

            refinedBoxes = tf.stack(BoxUtils.xywh_to_x0y0x1y1(x, y, w, h), axis=1)

            if needSizes:
                return refinedBoxes, refSizes, proposals_bboxs[:, 2:4]
            else:
                return refinedBoxes

    def getDisplacement(self, proposals):
        # proposals is first out in BoxregNet
        positives = self.roiMean(self.corr_feature, proposals)
        # fetch new data
        return positives