"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, NumberField, StringField, ListField
from ..representation import DetectionPrediction, ContainerPrediction
from ..utils import get_or_parse_value


class TinyYOLOv1Adapter(Adapter):
    """
    Class for converting output of Tiny YOLO v1 model to DetectionPrediction representation
    """
    __provider__ = 'tiny_yolo_v1'

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
             list of DetectionPrediction objects
        """
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]

        PROBABILITY_SIZE = 980
        CONFIDENCE_SIZE = 98
        BOXES_SIZE = 392

        CELLS_X, CELLS_Y = 7, 7
        CLASSES = 20
        OBJECTS_PER_CELL = 2

        result = []
        for identifier, output in zip(identifiers, prediction):
            assert PROBABILITY_SIZE + CONFIDENCE_SIZE + BOXES_SIZE == output.shape[0]

            probability, scale, boxes = np.split(output, [PROBABILITY_SIZE, PROBABILITY_SIZE + CONFIDENCE_SIZE])

            probability = np.reshape(probability, (CELLS_Y, CELLS_X, CLASSES))
            scale = np.reshape(scale, (CELLS_Y, CELLS_X, OBJECTS_PER_CELL))
            boxes = np.reshape(boxes, (CELLS_Y, CELLS_X, OBJECTS_PER_CELL, 4))

            confidence = np.zeros((CELLS_Y, CELLS_X, OBJECTS_PER_CELL, CLASSES + 4))
            for cls in range(CLASSES):
                confidence[:, :, 0, cls] = np.multiply(probability[:, :, cls], scale[:, :, 0])
                confidence[:, :, 1, cls] = np.multiply(probability[:, :, cls], scale[:, :, 1])

            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            for i, j, k in np.ndindex((CELLS_X, CELLS_Y, OBJECTS_PER_CELL)):
                box = boxes[j, i, k]
                box = [(box[0] + i) / float(CELLS_X), (box[1] + j) / float(CELLS_Y), box[2] ** 2, box[3] ** 2]

                label = np.argmax(confidence[j, i, k, :CLASSES])
                score = confidence[j, i, k, label]

                labels.append(label)
                scores.append(score)
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


PRECOMPUTED_ANCHORS = {
    'yolo_v2': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
    'tiny_yolo_v2': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
    'yolo_v3': [
        10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0
    ],
    'tiny_yolo_v3': [10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0]
}


def entry_index(w, h, n_coords, n_classes, pos, entry):
    row = pos // (w * h)
    col = pos % (w * h)
    return row * w * h * (n_classes + n_coords + 1) + entry * w * h + col


class BaseYoloAdapterConfig(ConfigValidator):
    classes = NumberField(floats=False, optional=True, min_value=1)
    coords = NumberField(floats=False, optional=True, min_value=1)
    num = NumberField(floats=False, optional=True, min_value=1)
    anchors = StringField(optional=True)


class YoloV2Adapter(Adapter):
    """
    Class for converting output of YOLO v2 family models to DetectionPrediction representation
    """
    __provider__ = 'yolo_v2'

    def validate_config(self):
        yolo_v2_adapter_config = BaseYoloAdapterConfig('BaseYoloAdapter_Config')
        yolo_v2_adapter_config.validate(self.launcher_config)

    def configure(self):
        self.classes = self.launcher_config.get('classes', 20)
        self.coords = self.launcher_config.get('coords', 4)
        self.num = self.launcher_config.get('num', 5)
        self.anchors = get_or_parse_value(self.launcher_config.get('anchors', 'yolo_v2'), PRECOMPUTED_ANCHORS)

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)[self.output_blob]

        cells_x, cells_y = 13, 13

        result = []
        for identifier, prediction in zip(identifiers, predictions):
            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            for y, x, n in np.ndindex((cells_y, cells_x, self.num)):
                index = n * cells_y * cells_x + y * cells_x + x

                box_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, 0)
                obj_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords)

                scale = prediction[obj_index]

                box = [
                    (x + prediction[box_index + 0 * (cells_y * cells_x)]) / cells_x,
                    (y + prediction[box_index + 1 * (cells_y * cells_x)]) / cells_y,
                    np.exp(prediction[box_index + 2 * (cells_y * cells_x)]) * self.anchors[2 * n + 0] / cells_x,
                    np.exp(prediction[box_index + 3 * (cells_y * cells_x)]) * self.anchors[2 * n + 1] / cells_y
                ]

                classes_prob = np.empty(self.classes)
                for cls in range(self.classes):
                    cls_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords + 1 + cls)
                    classes_prob[cls] = prediction[cls_index]

                classes_prob = classes_prob * scale

                label = np.argmax(classes_prob)

                labels.append(label)
                scores.append(classes_prob[label])
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


class YoloV3AdapterConfig(BaseYoloAdapterConfig):
    threshold = NumberField(floats=True, optional=True, min_value=0)
    outputs = ListField(optional=True)


class YoloV3Adapter(Adapter):
    """
    Class for converting output of YOLO v3 family models to DetectionPrediction representation
    """
    __provider__ = 'yolo_v3'

    def validate_config(self):
        yolo_v3_adapter_config = YoloV3AdapterConfig('YoloV3Adapter_Config')
        yolo_v3_adapter_config.validate(self.launcher_config)

    def configure(self):
        self.classes = self.launcher_config.get('classes', 80)
        self.coords = self.launcher_config.get('coords', 4)
        self.num = self.launcher_config.get('num', 3)
        self.anchors = get_or_parse_value(self.launcher_config.get('anchors', 'yolo_v3'), PRECOMPUTED_ANCHORS)
        self.threshold = self.launcher_config.get('threshold', 0.001)
        self.outputs = self.launcher_config.get('outputs', [])

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        def get_anchors_offset(x):
            return int((self.num * 2) * (len(self.anchors) / (self.num * 2) - 1 - math.log2(x / 13)))

        def parse_yolo_v3_results(prediction, threshold, w, h, det):
            cells_x, cells_y = prediction.shape[1:]
            prediction = prediction.flatten()
            for y, x, n in np.ndindex((cells_y, cells_x, self.num)):
                index = n * cells_y * cells_x + y * cells_x + x
                anchors_offset = get_anchors_offset(cells_x)

                box_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, 0)
                obj_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords)

                scale = prediction[obj_index]
                if scale < threshold:
                    continue

                box = [
                    (x + prediction[box_index + 0 * (cells_y * cells_x)]) / cells_x,
                    (y + prediction[box_index + 1 * (cells_y * cells_x)]) / cells_y,
                    np.exp(prediction[box_index + 2 * (cells_y * cells_x)]) * self.anchors[
                        anchors_offset + 2 * n + 0] / w,
                    np.exp(prediction[box_index + 3 * (cells_y * cells_x)]) * self.anchors[
                        anchors_offset + 2 * n + 1] / h
                ]

                classes_prob = np.empty(self.classes)
                for cls in range(self.classes):
                    cls_index = entry_index(cells_x, cells_y, self.coords, self.classes, index,
                                            self.coords + 1 + cls)
                    classes_prob[cls] = prediction[cls_index] * scale

                    det['labels'].append(cls)
                    det['scores'].append(classes_prob[cls])
                    det['x_mins'].append(box[0] - box[2] / 2.0)
                    det['y_mins'].append(box[1] - box[3] / 2.0)
                    det['x_maxs'].append(box[0] + box[2] / 2.0)
                    det['y_maxs'].append(box[1] + box[3] / 2.0)

            return det

        result = []

        raw_outputs = self._extract_predictions(raw, frame_meta)

        if self.outputs:
            outputs = self.outputs
        else:
            outputs = raw_outputs.keys()

        batch = len(identifiers)
        predictions = [[] for _ in range(batch)]
        for blob in outputs:
            for b in range(batch):
                predictions[b].append(raw[blob][b])

        for identifier, prediction, meta in zip(identifiers, predictions, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            input_shape = list(meta.get('input_shape', {'data': (3, 416, 416)}).values())[0]
            self.input_width = input_shape[2]
            self.input_height = input_shape[1]

            for p in prediction:
                parse_yolo_v3_results(p, self.threshold, self.input_width, self.input_height, detections)

            result.append(DetectionPrediction(
                identifier, detections['labels'], detections['scores'], detections['x_mins'], detections['y_mins'],
                detections['x_maxs'], detections['y_maxs']
            ))

        return result


class SSDAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd'

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        raw_outputs = self._extract_predictions(raw, frame_meta)
        prediction_batch = raw_outputs[self.output_blob]
        prediction_count = prediction_batch.shape[2]
        prediction_batch = prediction_batch.reshape(prediction_count, -1)
        prediction_batch = self.remove_empty_detections(prediction_batch)

        result = []
        for batch_index, identifier in enumerate(identifiers):
            prediction_mask = np.where(prediction_batch[:, 0] == batch_index)
            detections = prediction_batch[prediction_mask]
            detections = detections[:, 1::]
            result.append(DetectionPrediction(identifier, *zip(*detections)))

        return result

    @staticmethod
    def remove_empty_detections(prediction_blob):
        ind = prediction_blob[:, 0]
        ind_ = np.where(ind == -1)[0]
        m = ind_[0] if ind_.size else prediction_blob.shape[0]
        return prediction_blob[:m, :]


class FacePersonDetectionAdapterConfig(ConfigValidator):
    type = StringField()
    face_out = StringField()
    person_out = StringField()


class FacePersonAdapter(Adapter):
    __provider__ = 'face_person_detection'

    def validate_config(self):
        face_person_detection_adapter_config = FacePersonDetectionAdapterConfig(
            'FacePersonDetection_Config', on_extra_argument=FacePersonDetectionAdapterConfig.ERROR_ON_EXTRA_ARGUMENT)
        face_person_detection_adapter_config.validate(self.launcher_config)

    def configure(self):
        self.face_detection_out = self.launcher_config['face_out']
        self.person_detection_out = self.launcher_config['person_out']
        self.face_adapter = SSDAdapter(self.launcher_config, self.label_map, self.face_detection_out)
        self.person_adapter = SSDAdapter(self.launcher_config, self.label_map, self.person_detection_out)

    def process(self, raw, identifiers=None, frame_meta=None):
        face_batch_result = self.face_adapter(raw, identifiers)
        person_batch_result = self.person_adapter(raw, identifiers)
        result = [ContainerPrediction({self.face_detection_out: face_result, self.person_detection_out: person_result})
                  for face_result, person_result in zip(face_batch_result, person_batch_result)]

        return result
