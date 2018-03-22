
from datasets.process.DETRAC import FetchAll, loadImage
import cv2
import pickle
import numpy as np
HomeDir = "/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/ImageSets/Main/"
ImageDir = "/home/slh/dataset/ai/VOCdevkit_2CLS/VOC2007/JPEGImages/"
Train_set = "trainval.txt"
Test_set = "test.txt"
TrainImages = []
TrainAnns = {}
TestImages = []
TestAnns = {}
path = "/home/slh/dataset/DETRAC/"
bbox, Anns = FetchAll(path + "DETRAC-Train-Annotations-XML/list",
                      path + "DETRAC-Train-Annotations-XML/")
images = list(Anns.keys())
# training
# ftrain = open(HomeDir + Train_set, 'r')
# for item in ftrain:
#     # img = cv2.imread(ImageDir + item[:-1] + ".jpg")
#     ans = Anns[item[:-1]]
#     for index in ans:
#         if item[:-1] in TrainAnns.keys():
#             TrainAnns[item[:-1]].append(bbox[index])
#         else:
#             TrainAnns[item[:-1]] = [bbox[index]]
#
# ftrain.close()
# fw = open('TrainAns.pkl','wb')
# pickle.dump(TrainAnns, fw)
# fw.close()



# testing
ftrain = open(HomeDir + Test_set, 'r')
begin = 0
number = 120
done = False
medium = 1200
for item in ftrain:
    # img = cv2.imread(ImageDir + item[:-1] + ".jpg")
    if begin >= number and not done:
        fw = open('TestAns_120.pkl', 'wb')
        pickle.dump(TestAnns, fw)
        fw.close()
        done = True

    if begin >= medium:
        fw = open('TestAns_1200.pkl', 'wb')
        pickle.dump(TestAnns, fw)
        fw.close()
        break

    ans = Anns[item[:-1]]
    for index in ans:
        if item[:-1] in TestAnns.keys():
            TestAnns[item[:-1]].append(bbox[index])
        else:
            TestAnns[item[:-1]] = [bbox[index]]
    begin += 1

ftrain.close()
