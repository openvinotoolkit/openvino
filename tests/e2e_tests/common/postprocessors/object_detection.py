# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Object detection postprocessor."""
import logging as log

import numpy as np

from .provider import ClassProvider


class ParseBeforeODParser(ClassProvider):
    """Prepare the pipeline output to right state before parse_object_detection postprocessing using.
    e.g. (2,100,7) where all information from 2 batches contain in first element with shape (100, 7),
    other 'strings' in this element and second element is zeroes.
    Output transform to reference-like state: (n, 7) shape where 'n' is number of detections +  stop element('-1') """

    __action_name__ = "parse_before_OD"

    def __init__(self, config):
        self.target_layers = config.get("target_layers", None)
        pass

    def apply(self, data):
        """Parse data"""
        predictions = {}
        postprocessed = False
        target_layers = self.target_layers if self.target_layers else data.keys()
        for layer in target_layers:
            layer_data = np.squeeze(data[layer])
            assert layer_data.shape[-1] == 7, "Wrong data for postprocessing! Last dimension must be equal 7."
            if len(layer_data.shape) > 2:
                layer_data = np.reshape(layer_data, (-1, 7))
            predictions[layer] = layer_data
            postprocessed = True
        if not postprocessed:
            log.info("Postprocessor {} has nothing to process.".format(str(self.__action_name__)))
        return predictions


class ParseObjectDetection(ClassProvider):
    """Object detection parser."""
    __action_name__ = "parse_object_detection"

    def __init__(self, config):
        self.target_layers = config.get("target_layers", None)
        pass

    def apply(self, data):
        """Parse object detection data."""
        predictions = {}
        postprocessed = False
        target_layers = self.target_layers if self.target_layers else data.keys()
        dict_keys = ['class', 'prob', 'xmin', 'ymin', 'xmax', 'ymax']
        for layer in target_layers:
            predictions[layer] = []
            layer_data = np.squeeze(data[layer])
            # 1 detection leads to 0-d array after squeeze, which is not iterable
            if layer_data.ndim == 1:
                layer_data = np.expand_dims(layer_data, axis=0)
            assert len(layer_data.shape) <= 2, "Wrong data for postprocessing! Data length must be equal 2."
            for obj in layer_data:
                if type(obj) == np.float64:
                    log.debug(f"{obj} has type np.float64")
                    break
                elif obj[0] == -1:
                    log.debug(f"First item of {obj} == -1")
                    break
                assert len(obj) == 7, "Wrong data for postprocessing! Data length for one detection must be equal 7."
                while obj[0] > len(predictions[layer]) - 1:
                    predictions[layer].append([])
                box = dict(zip(dict_keys, obj[1:]))
                predictions[layer][int(obj[0])].append(box)
            postprocessed = True
        for layer in data.keys() - target_layers:
            predictions[layer] = data[layer]
        if postprocessed == False:
            log.info("Postprocessor {} has nothing to process".format(str(self.__action_name__)))
        return predictions


class ParseObjectDetectionTF(ClassProvider):
    """TF models yield 4-tensor format that needs to be converted into common format"""
    __action_name__ = "tf_to_common_od_format"

    def __init__(self, config):
        self.target_layers = ['num_detections', 'detection_classes',
                              'detection_scores', 'detection_boxes']

    def apply(self, data: dict):
        predictions = []
        num_batches = len(data['detection_boxes'])
        for b in range(num_batches):
            predictions.append([])
            num_detections = int(data['num_detections'][b])
            detection_classes = data['detection_classes'][b]
            detection_scores = data['detection_scores'][b]
            detection_boxes = data['detection_boxes'][b]
            for i in range(num_detections):
                obj = [
                    b, detection_classes[i], detection_scores[i],
                    detection_boxes[i][1], detection_boxes[i][0],
                    detection_boxes[i][3], detection_boxes[i][2]
                ]
                predictions[b].append(obj)
        predictions = np.asarray(predictions)
        if predictions.size != 0:
            predictions = np.reshape(predictions, newshape=(1, 1, predictions.shape[0] * predictions.shape[1],
                                                            predictions.shape[2]))
        else:
            log.error("Provided data doesn't contain any detected objects!")
        parsed_data = {'tf_detections': predictions}
        for layer, blob in data.items():
            if layer not in self.target_layers:
                parsed_data.update({layer: blob})
        return parsed_data


class ParseObjectDetectionMaskRCNN(ClassProvider):
    """TF models yield 4-tensor format that needs to be converted into common format"""
    __action_name__ = "parse_object_detection_mask_rcnn"

    def __init__(self, config):
        self.target_layers = ['num_detections', 'detection_classes',
                              'detection_scores', 'detection_boxes']

    def apply(self, data: dict):
        predictions = []
        num_batches = len(data['detection_boxes'])
        for b in range(num_batches):
            predictions.append([])
            num_detections = int(data['num_detections'][b])
            detection_classes = data['detection_classes'][b]
            detection_scores = data['detection_scores'][b]
            detection_boxes = data['detection_boxes'][b]
            for i in range(num_detections):
                obj = [
                    b, detection_classes[i], detection_scores[i],
                    detection_boxes[i][1], detection_boxes[i][0],
                    detection_boxes[i][3], detection_boxes[i][2]
                ]
                predictions[b].append(obj)
        parsed_data = {'tf_detections': np.array(predictions)}
        for layer, blob in data.items():
            if layer not in self.target_layers:
                parsed_data.update({layer: blob})
        return parsed_data


class AlignWithBatch(ClassProvider):
    """Batch alignment preprocessor.

    Duplicates 1-batch data BATCH number of times.
    """
    __action_name__ = "align_with_batch_od"

    def __init__(self, config):
        self.batch = config["batch"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply batch alignment (duplication) to data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        for layer in apply_to:

            container = np.zeros(shape=(1, 1, data[layer].shape[2] * self.batch + 1, data[layer].shape[3]))
            detections_counter = 0

            for b in range(self.batch):
                for box in data[layer][0][0]:
                    if box[0] == -1:
                        break
                    box[0] = b
                    container[0][0][detections_counter] = box
                    detections_counter += 1
            else:
                container[0][0][detections_counter] = [-1, 0, 0, 0, 0, 0, 0]  # Add 'stop' entry

            data[layer] = container

        return data


class ClipBoxes(ClassProvider):
    """
    Clip boxes coordinates to target height and width
    """
    __action_name__ = "clip_boxes"

    def __init__(self, config):
        self.normalized_boxes = config.get("normalized_boxes", True)
        self.max_h = 1 if self.normalized_boxes else config.get("max_h")
        self.max_w = 1 if self.normalized_boxes else config.get("max_w")
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        for layer in apply_to:
            for b in range(len(data[layer])):
                for i, box in enumerate(data[layer][b]):
                    data[layer][b][i].update({"xmax": min(box["xmax"], self.max_w) if box["xmax"] > 0 else 0,
                                              "xmin": max(box["xmin"], 0),
                                              "ymax": min(box["ymax"], self.max_h) if box["ymax"] > 0 else 0,
                                              "ymin": max(box["ymin"], 0)
                                              })
        return data


class AddClass(ClassProvider):
    """Adding class values postprocessor.

    Adds class key and its value to detection dictionaries.
    """
    __action_name__ = "add_class"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)
        self.class_value = config.get('class_value', 0)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers else data.keys()
        for layer in apply_to:
            for batch_num in range(len(data[layer])):
                for i in range(len(data[layer][batch_num])):
                    data[layer][batch_num][i]['class'] = self.class_value
        return data
