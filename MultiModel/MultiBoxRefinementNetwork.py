# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Roi.ROIPoolingWrapper import positionSensitiveRoiPooling
import math
import basic.RandomSelect
import utils.BoxUtils as BoxUtils
import basic.MultiGather as MultiGather
import Loss.Loss as Loss

class MultiBoxRefinementNetwork:
	POOL_SIZE=3

	def __init__(self, input, nCategories, downsample=16, offset=[32,32], hardMining=True, batch=8, reuse=False):
		self.downsample = downsample
		self.offset = offset
		self.reuse = reuse
		self.nCategories = nCategories
		self.classMaps = slim.conv2d(input, (self.POOL_SIZE**2)*(1+nCategories), 3, activation_fn=None, scope='classMaps', reuse=self.reuse)
		self.regressionMap = slim.conv2d(input, (self.POOL_SIZE**2)*4, 3, activation_fn=None, scope='regressionMaps',reuse=self.reuse)

		self.batch = batch
		self.classMapsSplit = tf.split(self.classMaps, self.batch, axis=0)
		self.regressionMapSplit = tf.split(self.regressionMap, self.batch, axis=0)

		self.hardMining=hardMining

		# self.number = number
		#Magic parameters.
		self.posIouTheshold = 0.5
		self.negIouThesholdHi = 0.5
		self.negIouThesholdLo = 0.1
		self.nTrainBoxes = 128
		self.nTrainPositives = 32
		self.falseValue = 0.0002

	def roiPooling(self, layer, boxes):
		return positionSensitiveRoiPooling(layer, boxes, offset=self.offset, downsample=self.downsample, roiSize=self.POOL_SIZE)

	def roiMean(self, layer, boxes):
		with tf.name_scope("roiMean"):
			return tf.reduce_mean(self.roiPooling(layer, boxes), axis=[1,2])

	def getBoxScores(self, boxes, number):
		with tf.name_scope("getBoxScores"):
			return self.roiMean(self.classMapsSplit[number], boxes)

	def classRefinementLoss(self, boxes, refs, number):
		with tf.name_scope("classRefinementLoss"):
			netScores = self.getBoxScores(boxes, number)
			refOnehot = tf.one_hot(refs, self.nCategories+1, on_value=1.0 - self.nCategories*self.falseValue, off_value=self.falseValue)
		
			return tf.nn.softmax_cross_entropy_with_logits(logits=netScores, labels=refOnehot)

	def refineBoxes(self, boxes, needSizes, number):
		with tf.name_scope("refineBoxes"):
			boxFineData = self.roiMean(self.regressionMapSplit[number], boxes)

			x,y,w,h = BoxUtils.x0y0x1y1_to_xywh(*tf.unstack(boxes, axis=1))
			x_rel, y_rel, w_rel, h_rel = tf.unstack(boxFineData, axis=1)

			if needSizes:
				refSizes = tf.stack([h,w], axis=1)

			x = x + x_rel * w
			y = y + y_rel * h

			w = w * tf.exp(w_rel)
			h = h * tf.exp(h_rel)

			refinedBoxes = tf.stack(BoxUtils.xywh_to_x0y0x1y1(x,y,w,h), axis=1)

			if needSizes:
				return refinedBoxes, refSizes, boxFineData[:,2:4]
			else:
				return refinedBoxes

	def boxRefinementLoss(self, boxes, refBoxes, number):
		with tf.name_scope("boxesRefinementLoss"):
			refinedBoxes, refSizes, rawSizes = self.refineBoxes(boxes, True, number)
			return Loss.boxRegressionLoss(refinedBoxes, rawSizes, refBoxes, refSizes), refinedBoxes

	def loss(self, proposals, refBoxes, refClasses, number):
		with tf.name_scope("BoxRefinementNetworkLoss"):
			tmp = []
			totalBox = []
			for item in proposals:
				tmp.append(tf.stop_gradient(item))
			proposals = tmp

			def getPosLoss(positiveBoxes, positiveRefIndices, nPositive, refbx, refcls, number):
				with tf.name_scope("getPosLoss"):
					positiveRefIndices =  tf.reshape(positiveRefIndices,[-1,1])

					positiveClasses, positiveRefBoxes = MultiGather.gather([refcls, refbx], positiveRefIndices)
					positiveClasses = tf.cast(tf.cast(positiveClasses,tf.int8) + 1, tf.uint8)

					if not self.hardMining:
						selected = basic.RandomSelect.randomSelectIndex(tf.shape(positiveBoxes)[0], nPositive)
						positiveBoxes, positiveClasses, positiveRefBoxes = \
							MultiGather.gather([positiveBoxes, positiveClasses, positiveRefBoxes], selected)

					boxLoss, boxes = self.boxRefinementLoss(positiveBoxes, positiveRefBoxes, number)
					totalBox.append(boxes)
					return tf.tuple([self.classRefinementLoss(positiveBoxes, positiveClasses, number) +
									 boxLoss, tf.shape(positiveBoxes)[0]])

			def getNegLoss(negativeBoxes, nNegative, number):
				with tf.name_scope("getNetLoss"):
					if not self.hardMining:
						negativeIndices = basic.RandomSelect.randomSelectIndex(tf.shape(negativeBoxes)[0], nNegative)
						negativeBoxes = tf.gather_nd(negativeBoxes, negativeIndices)

					return self.classRefinementLoss(negativeBoxes, tf.zeros(tf.stack([tf.shape(negativeBoxes)[0],1]), dtype=tf.uint8), number)

			def returnNullLoss(number):
				return tf.tuple([tf.constant(0.0, tf.float32), tf.constant(0, tf.int32)]), tf.constant([[0.0, 0.0, 0.0, 0.0]])

			def getRefinementLoss():
				with tf.name_scope("getRefinementLoss"):
					boxList, clsList = tf.unstack(refBoxes,axis=0), tf.unstack(refClasses,axis=0)
					totalLoss = []
					for pro, bx, cls, nm in zip(proposals, boxList, clsList, range(self.batch)):
						tm = tf.range(number[nm], dtype=tf.int32)
						bx = tf.gather(bx, tm)
						cls = tf.gather(cls, tm)

						iou = BoxUtils.iou(pro, bx)

						maxIou = tf.reduce_max(iou, axis=1)
						bestIou = tf.expand_dims(tf.cast(tf.argmax(iou, axis=1), tf.int32), axis=-1)

						#Find positive and negative indices based on their IOU
						posBoxIndices = tf.cast(tf.where(maxIou > self.posIouTheshold), tf.int32)
						negBoxIndices = tf.cast(tf.where(tf.logical_and(maxIou < self.negIouThesholdHi, maxIou > self.negIouThesholdLo)), tf.int32)

						#Split the boxes and references
						posBoxes, posRefIndices = MultiGather.gather([pro, bestIou], posBoxIndices)
						negBoxes = tf.gather_nd(pro, negBoxIndices)

						#Add GT boxes
						posBoxes = tf.concat([posBoxes,bx], 0)
						posRefIndices = tf.concat([posRefIndices, tf.reshape(tf.range(tf.shape(cls)[0]), [-1,1])], 0)

						#Call the loss if the box collection is not empty
						nPositive = tf.shape(posBoxes)[0]
						nNegative = tf.shape(negBoxes)[0]

						tmp_number = tf.cast(number[nm], dtype=tf.float32)
						if self.hardMining:
							posLoss = tf.cond(nPositive > 0,
												   lambda: getPosLoss(posBoxes, posRefIndices, 0, bx, cls, nm)[0],
												   lambda: tf.zeros((0,), tf.float32) )
							# posLoss = posLoss[0]
							negLoss = tf.cond(nNegative > 0, lambda: getNegLoss(negBoxes, 0, nm), lambda: tf.zeros((0,), tf.float32))
							# posLoss = posLoss[0]
							allLoss = tf.concat([posLoss, negLoss], 0)
							totalLoss.append( tf.cond(tf.shape(allLoss)[0]>0,
										   lambda: tf.reduce_mean(basic.MultiGather.gatherTopK(allLoss, self.nTrainBoxes)),
										   lambda: tf.constant(0.0)) )
							# totalBox.append(box)
						else:
							posLoss, posCount = tf.cond(nPositive > 0,
														lambda: getPosLoss(posBoxes, posRefIndices, self.nTrainPositives, bx, cls, nm),
														lambda: tf.constant(0.0) )
							# posLoss, posCount = posLoss[0], posLoss[1]
							negLoss = tf.cond(nNegative > 0,
											  lambda: getNegLoss(negBoxes, self.nTrainBoxes-posCount, nm),
											  lambda: tf.constant(0.0))

							nPositive = tf.cast(tf.shape(posLoss)[0], tf.float32)
							nNegative = tf.cond(nNegative > 0, lambda: tf.cast(tf.shape(negLoss)[0], tf.float32), lambda: tf.constant(0.0))

							totalLoss.append( (tf.reduce_mean(posLoss)*nPositive + tf.reduce_mean(negLoss)*nNegative)/(nNegative+nPositive) )
							# totalBox.append(box)
					return tf.reduce_mean(tf.stack(totalLoss,axis=0))

		return tf.cond(tf.shape(refBoxes)[0] > 0,
					   lambda: [getRefinementLoss(),totalBox],
					   lambda: [tf.constant(0.0), [tf.constant([0.0,0.0,0.0,0.0],dtype=tf.float32)]*self.batch])

	def getBoxes(self, proposals, proposal_scores, maxOutputs=30, nmsThreshold=0.3, scoreThreshold=0.8):
		if scoreThreshold is None:
			scoreThreshold = 0

		with tf.name_scope("getBoxes"):
			posRes = []
			scRes = []
			clsRes = []
			for pro, nm in zip(proposals, range(self.batch)):
				scores = tf.nn.softmax(self.getBoxScores(pro, nm))

				classes = tf.argmax(scores, 1)
				scores = tf.reduce_max(scores, axis=1)
				posIndices = tf.cast(tf.where(tf.logical_and(classes > 0, scores>scoreThreshold)), tf.int32)

				positives, scores, classes = MultiGather.gather([pro, scores, classes], posIndices)
				positives = self.refineBoxes(positives, False, nm)

				#Final NMS
				posIndices = tf.image.non_max_suppression(positives, scores, iou_threshold=nmsThreshold, max_output_size=maxOutputs)
				posIndices = tf.expand_dims(posIndices, axis=-1)
				positives, scores, classes = MultiGather.gather([positives, scores, classes], posIndices)

				classes = tf.cast(tf.cast(classes,tf.int32) - 1, tf.uint8)
				posRes.append(positives)
				scRes.append(scores)
				clsRes.append(classes)
			return posRes, scRes, clsRes