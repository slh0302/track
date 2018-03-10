import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import basic.RandomSelect
import utils.BoxUtils as BoxUtils
import Loss.Loss as Loss
import basic.MultiGather as MultiGather

class MultiRPN:
    def __init__(self, input, anchors=None, immediateSize=512, weightDecay=1e-5, inputDownscale=16, offset=[32, 32]):
        self.input = input
        self.anchors = anchors
        self.inputDownscale = inputDownscale
        self.offset = offset
        self.anchors = anchors if anchors is not None else self.makeAnchors([64, 128, 256, 512])
        print("Anchors: ", self.anchors)
        self.tfAnchors = tf.constant(self.anchors, dtype=tf.float32)

        self.hA = tf.reshape(self.tfAnchors[:, 0], [-1])
        self.wA = tf.reshape(self.tfAnchors[:, 1], [-1])

        self.nAnchors = len(self.anchors)

        self.positiveIouThreshold = 0.7
        self.negativeIouThreshold = 0.3
        self.regressionWeight = 1.0

        self.nBoxLosses = 256
        self.nPositiveLosses = 128

        # dimensions
        with tf.name_scope('dimension_info'):
            s = tf.shape(self.input)
            self.hIn = s[1]
            self.wIn = s[2]

        self.imageH = tf.cast(self.hIn * self.inputDownscale + self.offset[0] * 2, tf.float32)
        self.imageW = tf.cast(self.wIn * self.inputDownscale + self.offset[1] * 2, tf.float32)

        self.define(immediateSize, weightDecay)

    def define(self, immediateSize, weightDecay):
        with tf.name_scope('RPN'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weightDecay), padding='SAME'):
                # box prediction layers
                with tf.name_scope('NN'):
                    net = slim.conv2d(self.input, immediateSize, 3, activation_fn=tf.nn.relu)
                    scores = slim.conv2d(net, 2 * self.nAnchors, 1, activation_fn=None)
                    boxRelativeCoordinates = slim.conv2d(net, 4 * self.nAnchors, 1, activation_fn=None)

                # split coordinates
                x_raw, y_raw, w_raw, h_raw = tf.split(boxRelativeCoordinates, 4, axis=3)

                # Save raw box sizes for loss
                self.rawSizes = BoxUtils.mergeBoxDataList([w_raw, h_raw])

                # Convert NN outputs to BBox coordinates
                # x0y0 x1y1
                self.boxes = BoxUtils.nnToImageBoxesList(x_raw, y_raw, w_raw, h_raw, self.wA, self.hA, self.inputDownscale,
                                                     self.offset)

                # store the size of every box
                with tf.name_scope('box_sizes'):
                    boxSizes = tf.reshape(self.tfAnchors, [1, 1, 1, -1, 2])
                    boxSizes = tf.tile(boxSizes, tf.stack([1, self.hIn, self.wIn, 1, 1]))
                    tmp = tf.reshape(boxSizes, [-1, 2])
                    self.boxSizes = tf.stack([tmp, tmp], axis=0)

                # scores
                self.scores = tf.reshape(scores, [2, -1, 2])

    def genAllAnchors(self):
        with tf.name_scope('genAllAnchors'):
            z = tf.zeros([1, self.hIn, self.wIn, self.nAnchors], tf.float32)
            return BoxUtils.nnToImageBoxes(z, z, z, z, self.wA, self.hA, self.inputDownscale, self.offset)

    def loss(self, refBoxes):
        # refBoxes 2, ? , 4
        def getPositiveBoxes(boxes, refbx):
            with tf.name_scope('getPositiveBoxes'):
                iou = BoxUtils.iou(boxes, refbx)

                maxIou = tf.reduce_max(iou, axis=1)
                bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

                bestAnchors = tf.argmax(iou, axis=0)
                # Box matching matrix
                boxMatches = tf.cast(iou > self.positiveIouThreshold, tf.float32)

                boxMatches = tf.minimum(boxMatches + tf.transpose(tf.one_hot(bestAnchors, tf.shape(boxMatches)[0])),
                                        1.0)

                boxMatchMatrix = tf.stop_gradient(boxMatches)

                # Find positive boxes
                oneIfPositive = tf.reduce_max(boxMatchMatrix, axis=1)
                oneIfPositive = tf.stop_gradient(oneIfPositive)

                return oneIfPositive, maxIou, bestIou

        def getPositiveLoss(boxes, rawSizes, boxSizes, positiveIndices, bestIou, classificationLoss, refbx):
            with tf.name_scope('getPositiveLoss'):
                positiveBoxes, positiveRawSizes, positiveBoxSizes, positiveRefIndices, positiveClassificationLoss = \
                    MultiGather.gather([boxes, rawSizes, boxSizes, bestIou, classificationLoss], positiveIndices)

                # Regression loss
                positiveRefs = tf.gather_nd(refbx, positiveRefIndices)
                return Loss.boxRegressionLoss(positiveBoxes, positiveRawSizes, positiveRefs,
                                              positiveBoxSizes) * self.regressionWeight + positiveClassificationLoss

        def emptyPositiveLoss():
            with tf.name_scope('emptyPositiveLoss'):
                return tf.zeros((0,), tf.float32)

        def getNegativeLosses(boxes, negativeIndices, classificationLoss):
            with tf.name_scope('getNegativeLosses'):
                return tf.gather_nd(classificationLoss, negativeIndices)

        def emptyNegativeLoss():
            with tf.name_scope('emptyNegativeLoss'):
                return tf.zeros((0,), tf.float32)

        def calcAllLosses(refAnchors, boxes, rawSizes, scores, boxSizes, refbx):
            # refAnchors used to judge pos/neg boxes
            with tf.name_scope('calcAllRPNLosses'):
                tf.Print(refAnchors, [refAnchors])

                oneIfPositive, maxIou, bestIou = getPositiveBoxes(refAnchors, refbx)

                # Classification loss
                refScores = tf.one_hot(tf.cast(oneIfPositive > 0.5, tf.uint8), 2, on_value=0.999, off_value=0.001)
                classificationLoss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=refScores)

                # Split to positive and negative
                positiveIndices = tf.stop_gradient(tf.cast(tf.where(oneIfPositive >= 0.5), tf.int32))
                negativeIndices = tf.stop_gradient(
                    tf.cast(tf.where(tf.logical_and(oneIfPositive < 0.5, maxIou < self.negativeIouThreshold)),
                            tf.int32))
                tf.Print(positiveIndices, [positiveIndices])
                p = tf.cond(tf.shape(positiveIndices)[0] > 0,
                            lambda: getPositiveLoss(boxes, rawSizes, boxSizes, positiveIndices, bestIou,
                                                    classificationLoss, refbx), lambda: emptyPositiveLoss())
                n = tf.cond(tf.shape(negativeIndices)[0] > 0,
                            lambda: getNegativeLosses(boxes, negativeIndices, classificationLoss),
                            lambda: emptyNegativeLoss())

                # return positive losses, negative losses, positive boxes, positive reference indices, negative boxes
                return p, n


        def selectAndSum(losses, n):
            # random pick pos/neg loss value , totally n
            with tf.name_scope('selectAndSum'):
                l = basic.RandomSelect.randomSelectBatch(losses, n)
                return tf.reduce_mean(l)

        def calcLoss():
            with tf.name_scope('calcRPNLoss'):
                boxList, scoList, boxSizeList, rawList = tf.unstack(self.boxes,axis=0), tf.unstack(self.scores, axis=0), \
                                                         tf.unstack(self.boxSizes, axis=0), tf.unstack(self.rawSizes, axis=0)
                LossRes = []
                tf.Print(refBoxes, [refBoxes])
                for bx, sc, bxs, raws, refbx in zip(boxList, scoList, boxSizeList, rawList, refBoxes):
                    # Filter cross boundary boxes and get positives
                    inAnchros, inBoxes, inScores, inBoxSizes, inRawSizes = self.filterCrossBoundaryBoxes(
                        self.genAllAnchors(), [bx, sc, bxs, raws])
                    positiveLosses, negativeLosses = calcAllLosses(inAnchros, inBoxes, inRawSizes, inScores, inBoxSizes, refbx)

                    pCount = tf.shape(positiveLosses)[0]
                    nCount = tf.shape(negativeLosses)[0]

                    nPositive = tf.minimum(pCount, self.nPositiveLosses)
                    nNegative = tf.minimum(self.nBoxLosses - nPositive, nCount)
                    # choice n pos/neg loss, and sum
                    n = tf.cond(nNegative > 0, lambda: selectAndSum(negativeLosses, nNegative), lambda: tf.constant(0.0))
                    p = tf.cond(nPositive > 0, lambda: selectAndSum(positiveLosses, nPositive), lambda: tf.constant(0.0))
                    LossRes.append(n+p)

                return tf.reduce_mean(tf.stack(LossRes, axis=0))

        with tf.name_scope('RPNloss'):
            return tf.cond(tf.logical_and(tf.shape(refBoxes[0])[0] > 0,tf.shape(refBoxes[1])[0] > 0),
                           lambda: calcLoss(), lambda: tf.constant(0.0))

    def getInsideMask(self, boxes, boxInsideRate=1.0):
        # judge boxes whether in images
        with tf.name_scope('getInsideMask'):
            x0, y0, x1, y1 = tf.unstack(boxes, axis=1)

            if boxInsideRate != 1.0:
                w = x1 - x0 + 1.0
                h = y1 - y0 + 1.0

                xOutside = (1.0 - boxInsideRate) * w
                yOutside = (1.0 - boxInsideRate) * h
            else:
                xOutside = 0
                yOutside = 0

            return tf.logical_and(tf.logical_and(x0 >= -xOutside, y0 > -yOutside),
                                  tf.logical_and(x1 < (tf.cast(self.imageW, tf.float32) + xOutside),
                                                 y1 < (tf.cast(self.imageH, tf.float32) + yOutside)))

    def filterCrossBoundaryBoxes(self, boxes, others=[], boxInsideRate=1.0):
        with tf.name_scope('filterCrossBoundaryBoxes'):
            okIndices = tf.where(self.getInsideMask(boxes, boxInsideRate))
            okIndices = tf.cast(okIndices, tf.int32)

            return MultiGather.gather([boxes] + others, okIndices)

    def clipBoxesToEdge(self, boxes):
        with tf.name_scope("clipBoxesToEdge"):
            x0, y0, x1, y1 = tf.unstack(boxes, axis=1)
            x0 = tf.maximum(tf.minimum(x0, self.imageW), 0.0)
            x1 = tf.maximum(tf.minimum(x1, self.imageW), 0.0)
            y0 = tf.maximum(tf.minimum(y0, self.imageH), 0.0)
            y1 = tf.maximum(tf.minimum(y1, self.imageH), 0.0)
            return tf.stack([x0, y0, x1, y1], axis=1)

    def filterOutputBoxes(self, boxes, scores, others=[], preNmsCount=6000, maxOutSize=300, nmsThreshold=0.7):
        # boxes is self.boxes, init value in define()
        with tf.name_scope("filter_output_boxes"):
            boxList, scoreList = tf.unstack(boxes, axis=0), tf.unstack(scores, axis=0)
            boxRes = []
            scoRes = []
            for bx, sc in zip(boxList, scoreList):
                sc = tf.nn.softmax(sc)[:, 1]
                sc = tf.reshape(sc, [-1])

                # Clip boxes to edge
                bx = self.clipBoxesToEdge(bx)

                # Remove empty boxes
                bx, sc = BoxUtils.filterSmallBoxes(bx, [sc])
                sc, bx = tf.cond(tf.shape(sc)[0] > preNmsCount,
                                        lambda: tf.tuple(MultiGather.gatherTopK(sc, preNmsCount, [bx])),
                                        lambda: tf.tuple([sc, bx]))

                # NMS filter
                nmsIndices = tf.image.non_max_suppression(bx, sc, iou_threshold=nmsThreshold,
                                                          max_output_size=maxOutSize)
                nmsIndices = tf.expand_dims(nmsIndices, axis=-1)

                tmpBox, tmpScore = MultiGather.gather([bx, sc] + others, nmsIndices)

                # number is not equal
                boxRes.append(tmpBox)

                scoRes.append(tmpScore)

            return boxRes, scoRes

    def getPositiveOutputs(self, preNmsCount=6000, maxOutSize=300, nmsThreshold=0.7):
        # testing output
        boxes, scores = self.filterOutputBoxes(self.boxes, self.scores, preNmsCount=preNmsCount,
                                               nmsThreshold=nmsThreshold, maxOutSize=maxOutSize)
        return boxes, scores

    @staticmethod
    def makeAnchors(sizeList, sizeLim=[1024, 1024]):
        res = []
        for s in sizeList:
            res.append([s, s])
            if s * 2 <= sizeLim[0]:
                res.append([int(s * math.sqrt(2)), int(s / math.sqrt(2))])
            if s * 2 <= sizeLim[1]:
                res.append([int(s / math.sqrt(2)), int(s * math.sqrt(2))])

        return res
