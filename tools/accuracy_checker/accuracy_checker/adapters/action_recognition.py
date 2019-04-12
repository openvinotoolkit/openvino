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

import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, NumberField
from ..representation import DetectionPrediction, ContainerPrediction


class ActionDetectorConfig(ConfigValidator):
    type = StringField()
    priorbox_out = StringField()
    loc_out = StringField()
    main_conf_out = StringField()
    add_conf_out_prefix = StringField()
    add_conf_out_count = NumberField(optional=True, min_value=1)
    num_action_classes = NumberField()
    detection_threshold = NumberField(optional=True, floats=True, min_value=0, max_value=1)


class ActionDetection(Adapter):
    __provider__ = 'action_detection'

    def validate_config(self):
        action_detector_adapter_config = ActionDetectorConfig('ActionDetector_Config')
        action_detector_adapter_config.validate(self.launcher_config)

    def configure(self):
        self.priorbox_out = self.launcher_config['priorbox_out']
        self.loc_out = self.launcher_config['loc_out']
        self.main_conf_out = self.launcher_config['main_conf_out']
        self.num_action_classes = self.launcher_config['num_action_classes']
        self.detection_threshold = self.launcher_config.get('detection_threshold', 0)
        add_conf_out_count = self.launcher_config.get('add_conf_out_count')
        add_conf_out_prefix = self.launcher_config['add_conf_out_prefix']
        if add_conf_out_count is None:
            self.add_conf_outs = [add_conf_out_prefix]
        else:
            self.add_conf_outs = []
            for num in np.arange(start=1, stop=add_conf_out_count + 1):
                self.add_conf_outs.append('{}{}'.format(add_conf_out_prefix, num))

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        prior_boxes = raw_outputs[self.priorbox_out][0][0].reshape(-1, 4)
        prior_variances = raw_outputs[self.priorbox_out][0][1].reshape(-1, 4)
        for batch_id, identifier in enumerate(identifiers):
            labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores = self.prepare_detection_for_id(
                batch_id, raw_outputs, prior_boxes, prior_variances
            )
            action_prediction = DetectionPrediction(identifier, labels, class_scores, x_mins, y_mins, x_maxs, y_maxs)
            person_prediction = DetectionPrediction(
                identifier, [1] * len(labels), main_scores, x_mins, y_mins, x_maxs, y_maxs
            )
            result.append(ContainerPrediction({
                'action_prediction': action_prediction, 'class_agnostic_prediction': person_prediction
            }))

        return result

    def prepare_detection_for_id(self, batch_id, raw_outputs, prior_boxes, prior_variances):
        num_detections = raw_outputs[self.loc_out][batch_id].size // 4
        locs = raw_outputs[self.loc_out][batch_id].reshape(-1, 4)
        main_conf = raw_outputs[self.main_conf_out][batch_id].reshape(num_detections, -1)
        add_confs = list(map(
            lambda layer: raw_outputs[layer][batch_id].reshape(-1, self.num_action_classes), self.add_conf_outs
        ))
        anchors_num = len(add_confs)
        labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores = [], [], [], [], [], [], []
        for index in range(num_detections):
            if main_conf[index, 1] < self.detection_threshold:
                continue

            x_min, y_min, x_max, y_max = self.decode_box(prior_boxes[index], prior_variances[index], locs[index])
            action_confs = add_confs[index % anchors_num][index // anchors_num]
            action_label = np.argmax(action_confs)
            labels.append(action_label)
            class_scores.append(action_confs[action_label])
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)
            main_scores.append(main_conf[index, 1])

        return labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores

    @staticmethod
    def decode_box(prior, var, deltas):
        prior_width = prior[2] - prior[0]
        prior_height = prior[3] - prior[1]
        prior_center_x = (prior[0] + prior[2]) / 2
        prior_center_y = (prior[1] + prior[3]) / 2

        decoded_box_center_x = var[0] * deltas[0] * prior_width + prior_center_x
        decoded_box_center_y = var[1] * deltas[1] * prior_height + prior_center_y
        decoded_box_width = np.exp(var[2] * deltas[2]) * prior_width
        decoded_box_height = np.exp(var[3] * deltas[3]) * prior_height

        decoded_xmin = decoded_box_center_x - decoded_box_width / 2
        decoded_ymin = decoded_box_center_y - decoded_box_height / 2
        decoded_xmax = decoded_box_center_x + decoded_box_width / 2
        decoded_ymax = decoded_box_center_y + decoded_box_height / 2

        return decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax
