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

from pathlib import Path

from ..representation import DetectionAnnotation
from ..utils import get_key_by_value, read_json, read_xml

from .format_converter import FileBasedAnnotationConverter


class BITVehicleJSON(FileBasedAnnotationConverter):
    __provider__ = 'bitvehicle_json'

    def convert(self):
        annotations = []
        for annotation_image in read_json(self.annotation_file):
            labels, x_mins, y_mins, x_maxs, y_maxs, is_ignored, occluded = [], [], [], [], [], [], []
            for detection in annotation_image['objects']:
                x_min, y_min, x_max, y_max = detection['bbox']
                label = detection['label']

                if label == 'ignored':
                    for class_ in _CLASS_TO_IND.values():
                        is_ignored.append(len(labels))
                        labels.append(class_)
                        x_mins.append(x_min)
                        y_mins.append(y_min)
                        x_maxs.append(x_max)
                        y_maxs.append(y_max)
                else:
                    is_occluded = detection.get('is_occluded', False) or detection.get('occluded', False)
                    is_difficult = detection.get('difficult', False)
                    if is_occluded or is_difficult:
                        occluded.append(len(labels))

                    labels.append(_CLASS_TO_IND[label])
                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    x_maxs.append(x_max)
                    y_maxs.append(y_max)

            identifier = Path(annotation_image['image']).name
            annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
            annotation.metadata['is_occluded'] = occluded
            annotation.metadata['difficult_boxes'] = is_ignored

            annotations.append(annotation)

        return annotations, get_meta()


class BITVehicle(FileBasedAnnotationConverter):
    __provider__ = 'bitvehicle'

    def convert(self):
        annotations = []
        for annotation_image in read_xml(self.annotation_file):
            if annotation_image.tag != 'image':
                continue

            identifier = annotation_image.get('name')
            labels, x_mins, y_mins, x_maxs, y_maxs, occluded = [], [], [], [], [], []
            for roi in annotation_image.findall('box'):
                label = roi.get("label")
                x_left = int(roi.get('xtl'))
                x_right = int(roi.get('xbr'))
                y_top = int(roi.get('ytl'))
                y_bottom = int(roi.get('ybr'))
                x_min, y_min, x_max, y_max = x_left, y_top, x_right - x_left, y_bottom - y_top
                is_occluded = bool(int(roi.get('occluded')))

                labels.append(_CLASS_TO_IND[label])
                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)
                if is_occluded:
                    occluded.append(len(labels) - 1)

            annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
            annotation.metadata['is_occluded'] = occluded

            annotations.append(annotation)

        return annotations, get_meta()


_CLASSES = (
    '__background__',  # always index 0
    'vehicle',
    'plate'
)

_CLASS_TO_IND = dict(zip(_CLASSES, list(range(len(_CLASSES)))))


def get_meta():
    labels = dict(enumerate(_CLASSES))
    labels[-1] = 'ignored'

    return {'label_map': labels, 'background_label': get_key_by_value(labels, '__background__')}
