from utils.BoxLoader import BoxLoader
from datasets.car import CarDataset
dataset = BoxLoader()
test = CarDataset("/home/slh/dataset/DETRAC/", set="Train")
test.init()
da = test.load()
dataset.add(test)
