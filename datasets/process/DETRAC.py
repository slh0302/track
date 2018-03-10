
# -*- coding: utf8 -*-
import os
import cv2
import os.path as osp
import numpy as np
import json
import codecs
import xml.etree.ElementTree as ET
from lxml import etree
from pprint import pprint
XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

def parse_xml(fpath):

    assert fpath.endswith(XML_EXT), "Unsupport file format."

    parser = etree.XMLParser(encoding=ENCODE_METHOD)
    tree = ET.parse(fpath, parser=parser)
    root = tree.getroot()

    seq = {}
    seq['sequence_name'] = root.attrib['name']   # <sequence name="MVI_20011">

    # 1st children    <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
    seq['camera_state'] = root[0].attrib['camera_state']
    seq['sence_weather'] = root[0].attrib['sence_weather']

    # 2nd children    <ignored_region>
    ignored_regions = []
    for region in root[1]:
        xmin, ymin = int(round(float(region.attrib['left']))), int(round(float(region.attrib['top'])))
        width, height = int(round(float(region.attrib['width']))), int(round(float(region.attrib['height'])))
        ignore_region = [xmin, ymin, xmin +width, ymin +height]
        ignored_regions.append(ignore_region)
    seq['ignored_regions'] = ignored_regions

    # 3rd children    <frame>
    frames = {}
    for frame in root[2:]:
        restore_frame = {}
        restore_frame['frame_id'] = frame.attrib['num']
        restore_frame['frame_name'] = 'img%05d.jpg' % int(frame.attrib['num'])
        restore_frame['frame_density'] = frame.attrib['density']

        # target_list
        targets = {}
        for target in frame[0]:
            restore_target = {}
            restore_target['target_id'] = target.attrib['id']
            # box
            xmin, ymin = int(round(float(target[0].attrib['left']))), int(round(float(target[0].attrib['top'])))
            width, height = int(round(float(target[0].attrib['width']))), int(round(float(target[0].attrib['height'])))
            restore_target['target_bbox'] = [xmin, ymin, width, height]
            # attribute
            restore_target['target_orientation'] = target[1].attrib['orientation']
            restore_target['target_speed'] = target[1].attrib['speed']
            restore_target['target_trajectory_length'] = target[1].attrib['trajectory_length']
            restore_target['target_truncation_ratio'] = target[1].attrib['truncation_ratio']
            restore_target['target_vehicle_type'] = target[1].attrib['vehicle_type']
            targets[target.attrib['id']] = restore_target

        restore_frame['targets'] = targets
        frames['%s_img%05d' % (seq['sequence_name'], int(frame.attrib['num']))] = restore_frame

    seq['frames'] = frames
    return seq


def fetchSingleResult(xmlName):
    seq = parse_xml(xmlName)
    bboxes = {}
    begin = {}
    for frame_id in seq['frames']:
        targets = seq['frames'][frame_id]['targets']
        for tid in targets:
            target_id = targets[tid]['target_id']
            target_bbox = targets[tid]['target_bbox']
            if bboxes.has_key(target_id):
                bboxes[target_id].append(target_bbox)
            else:
                bboxes[target_id] = [target_bbox]
                begin[target_id] = [frame_id]

    return bboxes, begin

def FetchAll(xmllist, prefix=""):
    with open(xmllist, "r") as f:
        bboxes = []
        imgs = {}
        num = 0
        tragets = []
        for item in f:
            print(prefix + item[:-1])
            seq = parse_xml(prefix + item[:-1])
            for frame_id in seq['frames']:
                targets = seq['frames'][frame_id]['targets']
                for tid in targets:
                    # target_id = targets[tid]['target_id']
                    target_bbox = targets[tid]['target_bbox']
                    # can append other information
                    target_bbox.append(targets[tid]['target_id'])
                    target_bbox.append(targets[tid]['target_vehicle_type'])
                    if not targets[tid]['target_vehicle_type'] in tragets:
                        tragets.append(targets[tid]['target_vehicle_type'])
                    bboxes.append(target_bbox)
                    if frame_id in imgs.keys():
                        imgs[frame_id].append(num)
                    else:
                        imgs[frame_id] = [num]
                    num += 1
        print(tragets)
    return bboxes, imgs

def loadImage(img_id, prefix=""):
    sp = img_id.split('_')
    dir = '_'.join(sp[:-1])
    file = sp[-1] + ".jpg"
    return dir + "/" + file

def loadImages(begin_img_id, anns, prefix=""):
    sp = begin_img_id.split('_')
    dir = '_'.join(sp[:-1])
    file = sp[-1] + ".jpg"
    next_file = "img%05d" % (int(sp[-1][3:]) + 1)
    next_anns = ("%s_%s_%s" % (sp[0], sp[1], next_file))
    if not next_anns in anns.keys():
        next_file = "img%05d" % (int(sp[-1][3:]) - 1)
        next_anns = ("%s_%s_%s" % (sp[0], sp[1], next_file))
        nxet_dir = dir + "/" + next_file + ".jpg"
        return [nxet_dir, dir + "/" + file], [next_anns, begin_img_id]

    nxet_dir = dir + "/" + next_file + ".jpg"
    return [dir + "/" + file, nxet_dir], [begin_img_id, next_anns]

def FetchImages(Imagelist, prefix=""):
    with open(Imagelist, "r") as f:
        imgs = []
        for line in f:
            sp = line[:-1].split('/')
            image_id = '_'.join(sp[1:]).split('.')[0]
            imgs.append(image_id)
    return imgs

# if __name__ == '__main__':
#     bboxes, begin = FetchAll("/home/slh/tf-project/track/data/DETRAC-Train-Annotations-XML/list",
#                              "/home/slh/dataset/DETRAC/DETRAC-Train-Annotations-XML/")
#
#     st, ans = loadImages("MVI_20011_img01470", begin)
#     st1, ans1 = loadImages("MVI_20011_img01468", begin)
#     print(st)