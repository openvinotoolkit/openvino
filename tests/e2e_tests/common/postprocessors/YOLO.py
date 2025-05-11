# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

from .provider import ClassProvider

PRECOMPUTED_ANCHORS = {
    'yolo_v2': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
    'tiny_yolo_v2': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
    # TODO Understand why for tiny used 'yolo_v3' anchors
    'yolo_v3': [
        10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0
    ]
}


class YOLOV1Parser(ClassProvider):
    __action_name__ = "parse_yolo_V1_region"

    def __init__(self, config):
        self.classes = config['classes']
        self.coords = config['coords']
        self.num = config['num']
        self.grid = config['grid']

    def apply(self, prediction):
        probability_size = 980
        confidence_size = 98
        boxes_size = 392

        cells_x, cells_y = self.grid
        classes = self.classes
        objects_per_cell = self.num

        parsed_result = {}
        for layer, layer_data in prediction.items():
            parsed_result[layer] = []
            for b in range(layer_data.shape[0]):
                batch_data = []
                data = layer_data[b]
                assert probability_size + confidence_size + boxes_size == data.shape[0], "Wrong input data shape"

                prob, scale, boxes = np.split(data, [probability_size, probability_size + confidence_size])

                prob = np.reshape(prob, (cells_y, cells_x, classes))
                scale = np.reshape(scale, (cells_y, cells_x, objects_per_cell))
                boxes = np.reshape(boxes, (cells_y, cells_x, objects_per_cell, 4))

                probabilities = np.zeros((cells_y, cells_x, objects_per_cell, classes + 4))
                for cls in range(classes):
                    probabilities[:, :, 0, cls] = np.multiply(prob[:, :, cls], scale[:, :, 0])
                    probabilities[:, :, 1, cls] = np.multiply(prob[:, :, cls], scale[:, :, 1])

                for i, j, k in np.ndindex((cells_x, cells_y, objects_per_cell)):
                    box = boxes[j, i, k]
                    box = [(box[0] + i) / float(cells_x), (box[1] + j) / float(cells_y), box[2] ** 2, box[3] ** 2]

                    label = np.argmax(probabilities[j, i, k, :classes])
                    score = probabilities[j, i, k, label]
                    x_min = box[0] - box[2] / 2.0
                    y_min = box[1] - box[3] / 2.0
                    x_max = box[0] + box[2] / 2.0
                    y_max = box[1] + box[3] / 2.0

                    batch_data.append({"class": label, "xmin": x_min, "ymin": y_min,
                                       "xmax": x_max, "ymax": y_max, "prob": score})
                parsed_result[layer].append(batch_data)

        return parsed_result


class YOLOV2Parser(ClassProvider):
    __action_name__ = "parse_yolo_V2_region"

    def __init__(self, config):
        self.classes = config['classes']
        self.coords = config['coords']
        self.num = config['num']
        self.grid = config['grid']
        self.anchors = config.get('anchors', PRECOMPUTED_ANCHORS["yolo_v2"])
        self.scale_threshold = config.get('scale_threshold', 0.001)

    @staticmethod
    def _entry_index(w, h, n_coords, n_classes, pos, entry):
        row = pos // (w * h)
        col = pos % (w * h)
        return row * w * h * (n_classes + n_coords + 1) + entry * w * h + col

    @staticmethod
    def get_anchors_offset(x):
        return int(6 * (2 - (math.log2(x / 13))))

    def apply(self, data):
        parsed_result = {"yolo_v2_parsed": []}
        batches = max([l_data.shape[0] for l, l_data in data.items()])
        for b in range(batches):
            parsed_result["yolo_v2_parsed"].append([])
        for layer, layer_data in data.items():
            for b in range(layer_data.shape[0]):
                detections = layer_data[b]
                parsed = self._parse_yolo_v2_results(detections)
                parsed_result["yolo_v2_parsed"][b].extend(parsed)

        return parsed_result

    def _parse_yolo_v2_results(self, predictions):
        cells_x, cells_y = self.grid
        result = []

        for y, x, n in np.ndindex((cells_y, cells_x, self.num)):
            index = n * cells_y * cells_x + y * cells_x + x

            box_index = self._entry_index(cells_x, cells_y, self.coords, self.classes, index, 0)
            obj_index = self._entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords)

            scale = predictions[obj_index]

            box = [
                (x + predictions[box_index + 0 * (cells_y * cells_x)]) / cells_x,
                (y + predictions[box_index + 1 * (cells_y * cells_x)]) / cells_y,
                np.exp(predictions[box_index + 2 * (cells_y * cells_x)]) * self.anchors[2 * n + 0] / cells_x,
                np.exp(predictions[box_index + 3 * (cells_y * cells_x)]) * self.anchors[2 * n + 1] / cells_y
            ]

            classes_prob = np.empty(self.classes)
            for cls in range(self.classes):
                cls_index = self._entry_index(cells_x, cells_y, self.coords, self.classes, index,
                                              self.coords + 1 + cls)
                classes_prob[cls] = predictions[cls_index]

            classes_prob = classes_prob * scale

            label = np.argmax(classes_prob)
            score = classes_prob[label]
            x_min = box[0] - box[2] / 2.0
            y_min = box[1] - box[3] / 2.0
            x_max = box[0] + box[2] / 2.0
            y_max = box[1] + box[3] / 2.0

            result.append({"class": label, "xmin": x_min, "ymin": y_min,
                           "xmax": x_max, "ymax": y_max, "prob": score})
        return result


class YOLOV3Parser(ClassProvider):
    __action_name__ = "parse_yolo_V3_region"

    def __init__(self, config):
        self.classes = config['classes']
        self.coords = config['coords']
        self.masks_length = config['masks_length']
        self.input_w = config['input_w']
        self.input_h = config['input_h']
        self.scale_threshold = config.get('scale_threshold', 0.001)
        self.anchors = PRECOMPUTED_ANCHORS["yolo_v3"]

    @staticmethod
    def _entry_index(side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

    @staticmethod
    def get_anchors_offset(x):
        return int(6 * (2 - (math.log2(x / 13))))

    def _parse_yolo_v3_results(self, prediction):
        cells_x, cells_y = prediction.shape[1:]

        assert cells_y == cells_x, "Incorrect YOLO Region! Grid size sides are not equal"
        side = cells_x
        predictions = prediction.flatten()
        parsed_result = []

        side_square = cells_x * cells_y

        for i in range(side_square):
            row = i // side
            col = i % side
            for n in range(self.masks_length):
                obj_index = self._entry_index(side, self.coords, self.classes, n * side_square + i,
                                              self.coords)
                scale = predictions[obj_index]
                if scale < self.scale_threshold:
                    continue
                box_index = self._entry_index(side, self.coords, self.classes, n * side_square + i, 0)
                x = (col + predictions[box_index + 0 * side_square]) / side
                y = (row + predictions[box_index + 1 * side_square]) / side
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    w_exp = math.exp(predictions[box_index + 2 * side_square])
                    h_exp = math.exp(predictions[box_index + 3 * side_square])
                except OverflowError:
                    continue
                w = w_exp * self.anchors[self.get_anchors_offset(side) + 2 * n] / self.input_w
                h = h_exp * self.anchors[self.get_anchors_offset(side) + 2 * n + 1] / self.input_h

                for cls_id in range(self.classes):
                    class_index = self._entry_index(side, self.coords, self.classes, n * side_square + i,
                                                    self.coords + 1 + cls_id)
                    confidence = scale * predictions[class_index]

                    x_min = x - w / 2
                    y_min = y - h / 2
                    x_max = x_min + w
                    y_max = y_min + h

                    parsed_result.append({"class": cls_id, "xmin": x_min, "ymin": y_min,
                                   "xmax": x_max, "ymax": y_max, "prob": confidence})

        return parsed_result

    def apply(self, data):
        result = {"yolo_v3_parsed": []}
        batches = max([l_data.shape[0] for l, l_data in data.items()])
        for b in range(batches):
            result["yolo_v3_parsed"].append([])
        for layer, layer_data in data.items():
            for b in range(layer_data.shape[0]):
                detections = layer_data[b]
                parsed = self._parse_yolo_v3_results(detections)
                result["yolo_v3_parsed"][b].extend(parsed)

        return result


def logistic_activate(x):
    return 1. / (1. + math.exp(-x))


class YOLORegion(ClassProvider):
    __action_name__ = "yolo_region"

    def __init__(self, config):
        self.classes = config.get('classes')
        self.coords = config.get('coords')
        self.grid = config.get('grid')
        self.masks_length = config.get('masks_length', 3)
        self.do_softmax = bool(config.get("do_softmax", True))
        self.num = config.get('num') if self.do_softmax else self.masks_length

    @staticmethod
    def _entry_index(width, height, coords, classes, outputs, batch, location, entry):
        n = location // (width * height)
        loc = location % (width * height)
        return batch * outputs + n * width * height * (coords + classes + 1) + entry * width * height + loc
    @staticmethod
    def _logistic_activate(x):
        return 1. / (1. + math.exp(-x))
    @staticmethod
    def _softmax(data, B, C, H, W):
        dest_data = data.copy()
        for b in range(B):
            for i in range(H * W):
                max_val = data[b * C * H * W + i]
                for c in range(C):
                    val = data[b * C * H * W + c * H * W + i]
                    max_val = max(val, max_val)
                exp_sum = 0
                for c in range(C):
                    dest_data[b * C * H * W + c * H * W + i] = math.exp(data[b * C * H * W + c * H * W + i] - max_val)
                    exp_sum += dest_data[b * C * H * W + c * H * W + i]
                for c in range(C):
                    dest_data[b * C * H * W + c * H * W + i] = dest_data[b * C * H * W + c * H * W + i] / exp_sum
        return dest_data

    def apply(self, data):
        for layer, layer_data in data.items():

            B, C, IH, IW = layer_data.shape
            assert IH == IW, "Incorrect data layout! Input data should be in 'NCHW' format"

            if self.do_softmax:
                end_index = IW * IH
            else:
                end_index = IW * IH * (self.classes + 1)

            inputs_size = IH * IW * self.num * (self.classes + self.coords + 1)

            dst_data = layer_data.flatten()
            for b in range(B):
                for n in range(self.num):
                    index = self._entry_index(width=IW, height=IH, coords=self.coords, classes=self.classes,
                                              location=n * IW * IH, entry=0, outputs=inputs_size, batch=b)
                    for i in range(index, index + 2 * IW * IH):
                        dst_data[i] = self._logistic_activate(dst_data[i])

                    index = self._entry_index(width=IW, height=IH, coords=self.coords, classes=self.classes,
                                              location=n * IW * IH, entry=self.coords, outputs=inputs_size, batch=b)

                    for i in range(index, index + end_index):
                        dst_data[i] = self._logistic_activate(dst_data[i])

            if self.do_softmax:
                index = self._entry_index(IW, IH, self.coords, self.classes, inputs_size, 0, 0, self.coords + 1)
                batch_offset = inputs_size // self.num
                for b in range(B * self.num):
                    dst_data[index + b * batch_offset:] = self._softmax(data=dst_data[index + b * batch_offset:],
                                                                        B=1, C=self.classes, H=IH, W=IW)

            if self.do_softmax:
                data[layer] = dst_data.reshape((B, -1)) 
            else:
                data[layer] = dst_data.reshape((B, C, IH, IW))

        return data
