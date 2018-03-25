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


import random
import tensorflow as tf
import threading
import time
import numpy as np


class MultiTrackLoader:
    QUEUE_CAPACITY = 4

    def __init__(self, sources=[], initOnStart=True, batch=8):
        self.totalCount = 0
        self.counts = []
        self.sources = []
        self.initDone = False
        self.initOnStart = initOnStart
        self.batch = batch

        with tf.name_scope('dataset') as scope:
            self.queue = tf.FIFOQueue(dtypes=[tf.float32, tf.float32, tf.uint8, tf.int32, tf.float32, tf.float32, tf.int32],
                                      capacity=self.QUEUE_CAPACITY)

            self.image = tf.placeholder(dtype=tf.float32, shape=[self.batch, None, None, 3], name="image")
            self.boxes = tf.placeholder(dtype=tf.float32, shape=[self.batch, None, 4], name="boxes")
            self.classes = tf.placeholder(dtype=tf.uint8, shape=[self.batch, None], name="classes")
            self.number = tf.placeholder(dtype=tf.int32, shape=[self.batch], name="number")
            self.Disp1 = tf.placeholder(dtype=tf.float32, shape=[int(self.batch/2), None, 4], name="DispBoxes1")
            self.Disp2 = tf.placeholder(dtype=tf.float32, shape=[int(self.batch/2), None, 4], name="DispBoxes2")
            self.DispNumber = tf.placeholder(dtype=tf.int32, shape=[int(self.batch/2)], name="DispNumber")
            self.enqueueOp = self.queue.enqueue([self.image, self.boxes, self.classes, self.number, self.Disp1, self.Disp2, self.DispNumber])

        self.sources = sources[:]

    def categoryCount(self):
        return 80

    def threadFn(self, tid, sess):
        if tid == 0:
            if not self.initDone:
                self.init()
        else:
            while not self.initDone:
                time.sleep(1)

        while True:
            img, boxes, classes, Disp1, Disp2 = self.selectSource().load()
            boxes, number = self.paddingSame(boxes, pad=[0.0, 0.0, 0.0, 0.0])
            classes, _ = self.paddingSame(classes, pad=[100], length=number)
            Disp1, DispNumber = self.paddingSame(Disp1, pad=[0.0, 0.0, 0.0, 0.0])
            Disp2, _ = self.paddingSame(Disp2, pad=[[0.0, 0.0, 0.0, 0.0]], length=DispNumber)
            try:
                sess.run(self.enqueueOp,
                         feed_dict={self.image: img, self.boxes: boxes,
                                    self.classes: classes, self.number: number,
                                    self.Disp1: Disp1, self.Disp2: Disp2, self.DispNumber:DispNumber})

            except tf.errors.CancelledError:
                return

    def init(self):
        if not self.initOnStart:
            for s in self.sources:
                s.init()

        for s in self.sources:
            c = s.count()
            self.counts.append(c)
            self.totalCount += c

        print("BoxLoader: Loaded %d files." % self.totalCount)
        self.initDone = True

    def startThreads(self, sess, nThreads=4):
        self.threads = []
        for n in range(nThreads):
            t = threading.Thread(target=self.threadFn, args=(n, sess))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def get(self):
        images, boxes, classes, number, refDisp1, refDisp2, DispNumber = self.queue.dequeue()
        images.set_shape([self.batch, None, None, 3])
        boxes.set_shape([self.batch, None, 4])
        classes.set_shape([self.batch, None])
        number.set_shape([self.batch])
        refDisp1.set_shape([int(self.batch/2), None, 4])
        refDisp2.set_shape([int(self.batch / 2), None, 4])
        DispNumber.set_shape([int(self.batch / 2)])
        return images, boxes, classes, number, refDisp1, refDisp2, DispNumber

    def add(self, source):
        assert self.initDone == False
        if self.initOnStart:
            source.init()
        self.sources.append(source)

    def selectSource(self):
        i = random.randint(0, self.totalCount - 1)
        acc = 0
        for j in range(len(self.counts)):
            acc += self.counts[j]
            if acc >= i:
                return self.sources[j]

    def paddingSame(self, data, pad=[0.0], length=None):
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

    def paddingDispSame(self, data):
        pad = [0.0,0.0,0.0,0.0]
        tmp = [len(i) for i in data]
        maxLen = max(tmp)
        tmpList = []
        for index in range(len(data)):
            tmpList.append(data[index] + [pad] * (maxLen - tmp[index]))
        return tmpList, tmp


    def count(self):
        return self.totalCount

    def getCaptions(self, categories):
        return self.sources[0].getCaptions(categories)

    def getCaptionMap(self):
        return self.getCaptions(np.arange(0, self.categoryCount()))