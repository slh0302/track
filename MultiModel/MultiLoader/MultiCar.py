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
import numpy as np
import pickle
import cv2
import tensorflow as tf
from datasets.process import BoxAwareRandZoom
from datasets.process import DETRAC

class MultiCarDataset:
    def __init__(self, path, set="train", normalizeSize=True, randomZoom=False, batch=8):
        self.path = path
        self.car = None
        self.normalizeSize = normalizeSize
        self.set = set
        self.randomZoom = randomZoom
        self.batch = batch
        # 1
        self.caption = {"others":1, "car":1, "van":1, "bus":1}
        # 4
        # self.caption = {"others": 1, "car": 2, "van": 3, "bus": 4}
        # 80
        # self.caption = {"others": 3, "car": 3, "van": 8, "bus": 6}

    def init(self):
        # self.bbox, self.Anns = DETRAC.FetchAll(self.path + "DETRAC-Train-Annotations-XML/list",
        #                                        self.path + "DETRAC-Train-Annotations-XML/")
        f = open("/home/slh/tf-project/track/MultiModel/TFRecord/TestAns.pkl", 'rb')
        self.Anns = pickle.load(f)
        f.close()
        self.images = list(self.Anns.keys())
        print("Loaded " + str(len(self.images)) + " images")

    def getCaptions(self, categories):
        if categories is None:
            return None
        if not categories in self.caption.keys():
            return None
        return self.caption[categories]

    def load(self):
        while True:
            # imgId=self.images[1]
            # imgId=self.images[3456]
            imgs=[]
            boxList=[]
            idBox = []
            categorieList=[]
            # for
            for _ in range(int(self.batch / 2)):
                ids = random.randint(0, len(self.images) - 1)
                imgId = self.images[ids]

                #TODO for testing
                # data = ['MVI_20035_img00536', 'MVI_40871_img01596','MVI_39851_img00085']
                # ids = random.randint(0, 2)
                # imgId = data[ids]
                # pathList, annsList = DETRAC.loadImages(imgId, self.Anns)
                pathList, annsList = DETRAC.loadModImages(imgId, self.Anns)
                for path, ans in zip(pathList, annsList):
                    # imgFile = self.path + "Insight-MVT_Annotation_" + self.set + "/" + path
                    imgFile = "/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/" + path
                    img = cv2.imread(imgFile)

                    if img is None:
                        print("ERROR: Failed to load " + imgFile)
                        continue

                    sizeMul = 1.0
                    padTop = 0
                    padLeft = 0

                    if ans in self.Anns.keys():
                        instances = self.Anns[ans]

                    else:
                        # len(instances) <= 0:
                        continue

                    iBoxes = [{
                        "x": int(i[0]),
                        "y": int(i[1]),
                        "w": int(i[2]),
                        "h": int(i[3]),
                        "id": int(i[4])
                    } for i in instances]

                    if self.randomZoom:
                        img, iBoxes = BoxAwareRandZoom.randZoom(img, iBoxes, keepOriginalRatio=True, keepOriginalSize=True,
                                                                keepBoxes=True)

                    if self.normalizeSize:
                        sizeMul = 640.0 / min(img.shape[0], img.shape[1])
                        img = cv2.resize(img, (int(img.shape[1] * sizeMul), int(img.shape[0] * sizeMul)))

                    m = img.shape[1] % 32
                    if m != 0:
                        padLeft = int(m / 2)
                        img = img[:, padLeft: padLeft + img.shape[1] - m]

                    m = img.shape[0] % 32
                    if m != 0:
                        m = img.shape[0] % 32
                        padTop = int(m / 2)
                        img = img[padTop: padTop + img.shape[0] - m]

                    if img.shape[0] < 256 or img.shape[1] < 256:
                        print("Warning: Image to small, skipping: " + str(img.shape))
                        continue

                    boxes = []
                    idBoxes = {}
                    categories = []
                    for i in range(len(instances)):
                        x1, y1, w, h = iBoxes[i]["x"], iBoxes[i]["y"], iBoxes[i]["w"], iBoxes[i]["h"]
                        newBox = [int(x1 * sizeMul) - padLeft, int(y1 * sizeMul) - padTop, int((x1 + w) * sizeMul) - padLeft,
                                  int((y1 + h) * sizeMul) - padTop]
                        newBox[0] = max(min(newBox[0], img.shape[1]), 0)
                        newBox[1] = max(min(newBox[1], img.shape[0]), 0)
                        newBox[2] = max(min(newBox[2], img.shape[1]), 0)
                        newBox[3] = max(min(newBox[3], img.shape[0]), 0)
                        if (newBox[2] - newBox[0]) >= 16 and (newBox[3] - newBox[1]) >= 16:
                            boxes.append(newBox)
                            idBoxes[iBoxes[i]["id"]] = newBox
                            categories.append(self.caption[instances[i][-1]])

                    if len(boxes) == 0:
                        print("Warning: No boxes on image. Skipping.")
                        continue

                    boxes = np.array(boxes, dtype=np.float32)
                    boxes = np.reshape(boxes, [-1, 4]).tolist()
                    categories = np.array(categories, dtype=np.uint8).tolist()
                    imgs.append(img)
                    boxList.append(boxes)
                    idBox.append(idBoxes)
                    categorieList.append(categories)
            return imgs, boxList, categorieList# , pathList, annsList, ids
            # refDisp = self.compareID(idBox[0], idBox[1])
            # if len(refDisp[0]) == 0:
            #     continue
            # else:
            #     return imgs, boxList, categorieList, refDisp


    def compareID(self, box1, box2):
        intersection = list(set(box1.keys()).intersection(set(box2.keys())))
        refd1 = []
        refd2 = []
        for item in intersection:
            refd1.append(box1[item])
            refd2.append(box2[item])
        return [refd1, refd2]

    def count(self):
        return len(self.images)
