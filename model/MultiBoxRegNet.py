import tensorflow as tf
import tensorflow.contrib.slim as slim
from Roi import positionSensitiveRoiPooling
import math
import basic.RandomSelect
import utils.BoxUtils as BoxUtils
import basic.MultiGather as MultiGather
import Loss.Loss as Loss


class MultiBoxRegNet:
    POOL_SIZE = 3

    def __init__(self, input, nCategories, downsample=16, offset=[32, 32], hardMining=True):
        self.downsample = downsample
        self.offset = offset
        self.nCategories = nCategories
        self.classMaps = slim.conv2d(input, (self.POOL_SIZE ** 2) * (1 + nCategories), 3, activation_fn=None,
                                     scope='classMaps')
        self.splitClassMaps = tf.split(self.classMaps, 2, 0)
        self.regressionMap = slim.conv2d(input, (self.POOL_SIZE ** 2) * 4, 3, activation_fn=None,
                                         scope='regressionMaps')
        self.splitRegressionMaps = tf.split(self.regressionMap, 2, 0)
        self.hardMining = hardMining
        # self.testRefine = [tf.shape(self.splitClassMaps[0])]

        # Magic parameters.
        self.posIouTheshold = 0.5
        self.negIouThesholdHi = 0.5
        self.negIouThesholdLo = 0.1
        self.nTrainBoxes = 128
        self.nTrainPositives = 32
        self.falseValue = 0.0002

    def roiPooling(self, layer, boxes):
        return positionSensitiveRoiPooling(layer, boxes, offset=self.offset, downsample=self.downsample,
                                           roiSize=self.POOL_SIZE)

    def roiMean(self, layer, boxes):
        with tf.name_scope("roiMean"):
            return tf.reduce_mean(self.roiPooling(layer, boxes), axis=[1, 2])

    def getBoxScores(self, layer, boxes):
        with tf.name_scope("getBoxScores"):
            return self.roiMean(self.splitClassMaps[layer], boxes)

    def classRefinementLoss(self, boxes, refs, numLayer):
        with tf.name_scope("classRefinementLoss"):
            netScores = self.getBoxScores(numLayer, boxes)
            refOnehot = tf.one_hot(refs, self.nCategories + 1, on_value=1.0 - self.nCategories * self.falseValue,
                                   off_value=self.falseValue)

            return tf.nn.softmax_cross_entropy_with_logits(logits=netScores, labels=refOnehot)

    def refineBoxes(self, boxes, needSizes, numLayer):
        with tf.name_scope("refineBoxes"):
            boxFineData = self.roiMean(self.splitRegressionMaps[numLayer], boxes)
            # anchor boxes xy,w,h
            x, y, w, h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(boxes, axis=1))
            x_rel, y_rel, w_rel, h_rel = tf.unstack(boxFineData, axis=1)

            if needSizes:
                refSizes = tf.stack([h, w], axis=1)

            x = x + x_rel * w
            y = y + y_rel * h

            w = w * tf.exp(w_rel)
            h = h * tf.exp(h_rel)

            refinedBoxes = tf.stack(BoxUtils.xywh_to_x0y0x1y1(x, y, w, h), axis=1)

            if needSizes:
                return refinedBoxes, refSizes, boxFineData[:, 2:4]
            else:
                return refinedBoxes

    def boxRefinementLoss(self, boxes, refBoxes, numLayer):
        with tf.name_scope("boxesRefinementLoss"):
            refinedBoxes, refSizes, rawSizes = self.refineBoxes(boxes, True, numLayer)
            return Loss.boxRegressionLoss(refinedBoxes, rawSizes, refBoxes, refSizes), refinedBoxes

    def loss(self, proposals, refBoxes, refClasses):
        with tf.name_scope("BoxRefinementNetworkLoss"):
            tmp = []
            totalBoxes = []
            for item in proposals:
                item = tf.stop_gradient(item)
                tmp.append(item)
            proposals = tmp

            def getPosLoss(positiveBoxes, positiveRefIndices, nPositive, refcls, refbx, numLayer):
                with tf.name_scope("getPosLoss"):
                    positiveRefIndices = tf.reshape(positiveRefIndices, [-1, 1])

                    positiveClasses, positiveRefBoxes = MultiGather.gather([refcls, refbx], positiveRefIndices)
                    positiveClasses = tf.cast(tf.cast(positiveClasses, tf.int8) + 1, tf.uint8)

                    if not self.hardMining:
                        selected = basic.RandomSelect.randomSelectIndex(tf.shape(positiveBoxes)[0], nPositive)
                        positiveBoxes, positiveClasses, positiveRefBoxes = MultiGather.gather(
                            [positiveBoxes, positiveClasses, positiveRefBoxes], selected)
                    loss, box = self.boxRefinementLoss(
                        positiveBoxes, positiveRefBoxes, numLayer)
                    return tf.tuple([self.classRefinementLoss(positiveBoxes, positiveClasses, numLayer) + loss, tf.shape(positiveBoxes)[0]]), box

            def getNegLoss(negativeBoxes, nNegative, numLayer):
                with tf.name_scope("getNetLoss"):
                    if not self.hardMining:
                        negativeIndices = basic.RandomSelect.randomSelectIndex(tf.shape(negativeBoxes)[0], nNegative)
                        negativeBoxes = tf.gather_nd(negativeBoxes, negativeIndices)

                    return self.classRefinementLoss(negativeBoxes,
                                                    tf.zeros(tf.stack([tf.shape(negativeBoxes)[0], 1]), dtype=tf.uint8), numLayer)

            def returnNullLoss():
                return tf.tuple([tf.constant(0.0), tf.constant(0, tf.int32)]), tf.constant([[0.0,0.0,0.0,0.0]])

            def getRefinementLoss():
                with tf.name_scope("getRefinementLoss"):
                    totalLoss = []

                    # tf.Print(proposals, [tf.shape(proposals)])
                    for pro, bx, cls, num in zip(proposals, refBoxes, refClasses, range(0,2)):
                        iou = BoxUtils.iou(pro, bx)

                        maxIou = tf.reduce_max(iou, axis=1)
                        bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

                        # Find positive and negative indices based on their IOU
                        posBoxIndices = tf.cast(tf.where(maxIou > self.posIouTheshold), tf.int32)
                        negBoxIndices = tf.cast(
                            tf.where(tf.logical_and(maxIou < self.negIouThesholdHi, maxIou > self.negIouThesholdLo)),
                            tf.int32)

                        # Split the boxes and references
                        posBoxes, posRefIndices = MultiGather.gather([pro, bestIou], posBoxIndices)
                        negBoxes = tf.gather_nd(pro, negBoxIndices)

                        # Add GT boxes
                        posBoxes = tf.concat([posBoxes, bx], 0)
                        posRefIndices = tf.concat([posRefIndices, tf.reshape(tf.range(tf.shape(cls)[0]), [-1, 1])],0)

                        # Call the loss if the box collection is not empty
                        nPositive = tf.shape(posBoxes)[0]
                        nNegative = tf.shape(negBoxes)[0]

                        if self.hardMining:
                            posLoss, box = tf.cond(nPositive > 0, lambda: getPosLoss(posBoxes, posRefIndices, 0, cls, bx, num)[0],
                                              lambda: [tf.zeros((0,), tf.float32), tf.constant([[0,0,0,0]])])
                            negLoss = tf.cond(nNegative > 0, lambda: getNegLoss(negBoxes, 0, num),
                                              lambda: tf.zeros((0,), tf.float32))

                            allLoss = tf.concat([posLoss, negLoss], 0)
                            totalLoss.append( tf.cond(tf.shape(allLoss)[0] > 0,
                                           lambda: tf.reduce_mean(basic.MultiGather.gatherTopK(allLoss, self.nTrainBoxes)),
                                           lambda: tf.constant(0.0)) )
                            totalBoxes.append(box)
                        else:
                            posLoss, box = tf.cond(nPositive > 0,
                                                        lambda: getPosLoss(posBoxes, posRefIndices, self.nTrainPositives, cls, bx, num),
                                                        lambda: returnNullLoss() )
                            posCount = posLoss[1]
                            posLoss  = posLoss[0]

                            negLoss = tf.cond(nNegative > 0, lambda: getNegLoss(negBoxes, self.nTrainBoxes - posCount, num),
                                              lambda: tf.constant(0.0))

                            nPositive = tf.cast(tf.shape(posLoss)[0], tf.float32)
                            nNegative = tf.cond(nNegative > 0, lambda: tf.cast(tf.shape(negLoss)[0], tf.float32),
                                                lambda: tf.constant(0.0))

                            totalLoss.append( (tf.reduce_mean(posLoss) * nPositive + tf.reduce_mean(negLoss) * nNegative) / (
                                    nNegative + nPositive) )
                            totalBoxes.append(box)

                    return tf.reduce_mean(tf.stack(totalLoss, axis=0))

        return tf.cond(tf.logical_and(tf.shape(proposals[0])[0] > 0, tf.shape(refBoxes[0])[0] > 0),
                       lambda: getRefinementLoss(), lambda: tf.constant(0.0)), totalBoxes

    def getBoxes(self, proposals, proposal_scores, maxOutputs=30, nmsThreshold=0.3, scoreThreshold=0.8):
        if scoreThreshold is None:
            scoreThreshold = 0

        with tf.name_scope("getBoxes"):
            posRes = []
            scoRes = []
            clsRes = []
            self.debug_box = []
            for item, num in zip(proposals, range(0,2)):
                scores = tf.nn.softmax(self.getBoxScores(num, item))

                classes = tf.argmax(scores, 1)
                scores = tf.reduce_max(scores, axis=1)
                posIndices = tf.cast(tf.where(tf.logical_and(classes > 0, scores > scoreThreshold)), tf.int32)
                self.debug_box.append(scores)
                positives, scores, classes = MultiGather.gather([item, scores, classes], posIndices)
                positives = self.refineBoxes(positives, False, num)
                self.debug_box.append(tf.shape(positives))
                # Final NMS
                posIndices = tf.image.non_max_suppression(positives, scores, iou_threshold=nmsThreshold,
                                                          max_output_size=maxOutputs)
                posIndices = tf.expand_dims(posIndices, axis=-1)
                positives, scores, classes = MultiGather.gather([positives, scores, classes], posIndices)

                classes = tf.cast(tf.cast(classes, tf.int32) - 1, tf.uint8)
                posRes.append(positives)
                scoRes.append(scores)
                clsRes.append(classes)

            return posRes, scoRes, clsRes