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

from ..representation import DetectionPrediction
from ..adapters import Adapter


class XML2DetectionAdapter(Adapter):
    """
    Class for converting xml detection results in OpenCV FileStorage format to DetectionPrediction representation.
    """

    __provider__ = 'xml_detection'

    def process(self, tree, identifiers=None, frame_meta=None):
        class_to_ind = dict(zip(self.label_map.values(), range(len(self.label_map.values()))))

        result = {}
        for frames in tree.getroot():
            for frame in frames:
                identifier = frame.tag + '.png'
                labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
                for prediction in frame:
                    if prediction.find('is_ignored'):
                        continue

                    label = prediction.find('type')
                    if not label:
                        raise ValueError('Detection predictions contains detection without "{}"'.format('type'))
                    label = class_to_ind[label.text]

                    confidence = prediction.find('confidence')
                    if confidence is None:
                        raise ValueError('Detection predictions contains detection without "{}"'.format('confidence'))
                    confidence = float(confidence.text)

                    box = prediction.find('roi')
                    if not box:
                        raise ValueError('Detection predictions contains detection without "{}"'.format('roi'))
                    box = list(map(float, box.text.split()))

                    labels.append(label)
                    scores.append(confidence)
                    x_mins.append(box[0])
                    y_mins.append(box[1])
                    x_maxs.append(box[0] + box[2])
                    y_maxs.append(box[1] + box[3])

                    result[identifier] = DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs)

        return result
