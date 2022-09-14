"""
"""

import os
import cv2
import re
import numpy as np



def kitti2yolo(kitti_label_arr, img_height, img_width):

    x1 = kitti_label_arr[0]
    y1 = kitti_label_arr[1]
    x2 = kitti_label_arr[2]
    y2 = kitti_label_arr[3]

    bb_width = x2 - x1
    bb_height = y2 - y1
    yolo_x = (x1 + 0.5*bb_width) / img_width
    yolo_y = (y1 + 0.5*bb_height) / img_height
    yolo_bb_width = bb_width / img_width
    yolo_bb_height = bb_height / img_height

    return [yolo_x, yolo_y, yolo_bb_width, yolo_bb_height]


def read_labels(path, seq_id, classes, convert_to_yolo=True):
    labels = {}
    images = {}
    with open(os.path.join(path, 'label_02', f'{seq_id}.txt')) as f:
        for line in f.readlines():
            line = line.split(' ')
            img_id = line[0]
            cvimage_path = os.path.join(path, 'image_02', seq_id, f'{str(img_id).zfill(6)}.png')
            if convert_to_yolo:
                cvimage = cv2.imread(cvimage_path)
                height, width, frame_depth = cvimage.shape
            
            this_name = line[2]
            obj_id = line[1]

            if this_name in classes:
                bbox = [float(x) for x in line[6:10]]
                
                if convert_to_yolo:
                    bbox = kitti2yolo(bbox, height, width)
            
                if obj_id not in labels:
                    labels[obj_id] = []
                    images[obj_id] = []

                labels[obj_id].append(bbox)
                images[obj_id].append(cvimage_path)
    return labels, images


def read_image(path):
    return cv2.imread(path)