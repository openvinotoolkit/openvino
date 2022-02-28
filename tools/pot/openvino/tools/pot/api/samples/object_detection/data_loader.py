# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import cv2
import numpy as np

from openvino.tools.pot import DataLoader


class COCOLoader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.images_path = self.config.images_path
        self.annotation_path = self.config.annotation_path
        self.images = os.listdir(self.images_path)
        self.labels = None
        self.data, self.bbox = self.prepare_annotation()

    def prepare_annotation(self):
        with open(self.annotation_path) as f:
            file = json.load(f)
        self.labels = [i['id'] for i in file['categories']]
        data = {}
        for idx, image in enumerate(file['images']):
            data[idx] = {'file_name': image['file_name'], 'image_id': image['id'],
                         'height': image['height'], 'width': image['width']}
        bbox = {}
        for i in file['annotations']:
            if i['image_id'] not in bbox.keys():
                bbox[i['image_id']] = {'bbox': [self.prepare_bbox(*i['bbox'])],
                                       'category_id': [i['category_id']],
                                       'iscrowd': [i['iscrowd']]}
            else:
                bbox[i['image_id']]['bbox'].append(self.prepare_bbox(*i['bbox']))
                bbox[i['image_id']]['category_id'].append(i['category_id'])
                bbox[i['image_id']]['iscrowd'].append(i['iscrowd'])

        return data, bbox

    def __getitem__(self, index):
        """ Returns (img_id, img_annotation), image"""
        if index >= len(self):
            raise IndexError

        shape_image = self.data[index]['height'], self.data[index]['width']

        bbox = np.array(self.bbox.get(self.data[index]['image_id'], {}).get('bbox', []))

        if bbox.size != 0:
            x_maxs = np.max(bbox, axis=1)
            y_maxs = np.max(bbox, axis=1)
            x_mins = np.min(bbox, axis=1)
            y_mins = np.min(bbox, axis=1)
        else:
            x_maxs, y_maxs, x_mins, y_mins = [], [], [], []

        labels = np.array(self.bbox.get(self.data[index]['image_id'], {}).get('category_id', []))
        iscrowd = np.array(self.bbox.get(self.data[index]['image_id'], {}).get('iscrowd', []))

        annotation = {'boxes': bbox, 'labels': labels, 'iscrowd': iscrowd,
                      'x_maxs': x_maxs, 'x_mins': x_mins, 'y_maxs': y_maxs, 'y_mins': y_mins}
        annotation = [annotation, shape_image]
        return self._read_and_preprocess_image(self.images_path + self.data[index]['file_name']), annotation

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _read_and_preprocess_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (640, 640))
        return image

    @staticmethod
    def prepare_bbox(x, y, weight, height):
        return np.array([x, y, x + weight, y + height])
