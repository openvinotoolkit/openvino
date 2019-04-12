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
from ..config import PathField, NumberField
from ..representation import DetectionAnnotation
from ..utils import convert_bboxes_xywh_to_x1y1x2y2, read_xml, read_txt

from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class DetectionOpenCVConverterConfig(BaseFormatConverterConfig):
    annotation_file = PathField()
    image_names_file = PathField(optional=True)
    label_start = NumberField(floats=False, optional=True)
    background_label = NumberField(floats=False, optional=True)


class DetectionOpenCVStorageFormatConverter(BaseFormatConverter):
    __provider__ = 'detection_opencv_storage'

    _config_validator_type = DetectionOpenCVConverterConfig

    def configure(self):
        self.annotation_file = self.config['annotation_file']
        self.image_names_file = self.config.get('image_names_file')
        self.label_start = self.config.get('label_start', 1)
        self.background_label = self.config.get('background_label')

    def convert(self):
        root = read_xml(self.annotation_file)

        labels_set = self.get_label_set(root)

        labels_set = sorted(labels_set)
        class_to_ind = dict(zip(labels_set, list(range(self.label_start, len(labels_set) + self.label_start + 1))))
        label_map = {}
        for class_label, ind in class_to_ind.items():
            label_map[ind] = class_label

        annotations = []
        for frames in root:
            for frame in frames:
                identifier = '{}.png'.format(frame.tag)
                labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
                difficult_indices = []
                for annotation in frame:
                    label = annotation.findtext('type')
                    if not label:
                        raise ValueError('"{}" contains detection without "{}"'.format(self.annotation_file, 'type'))

                    box = annotation.findtext('roi')
                    if not box:
                        raise ValueError('"{}" contains detection without "{}"'.format(self.annotation_file, 'roi'))
                    box = list(map(float, box.split()))

                    is_ignored = annotation.findtext('is_ignored', 0)
                    if int(is_ignored) == 1:
                        difficult_indices.append(len(labels))

                    labels.append(class_to_ind[label])
                    x_min, y_min, x_max, y_max = convert_bboxes_xywh_to_x1y1x2y2(*box)
                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    x_maxs.append(x_max)
                    y_maxs.append(y_max)

                detection_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
                detection_annotation.metadata['difficult_boxes'] = difficult_indices
                annotations.append(detection_annotation)

        if self.image_names_file:
            self.rename_identifiers(annotations, self.image_names_file)

        meta = {}
        if self.background_label:
            label_map[self.background_label] = '__background__'
            meta['background_label'] = self.background_label
        meta['label_map'] = label_map

        return annotations, meta

    @staticmethod
    def rename_identifiers(annotation_list, images_file):
        for annotation, image in zip(annotation_list, read_txt(images_file)):
            annotation.identifier = image

        return annotation_list


    @staticmethod
    def get_label_set(xml_root):
        labels_set = set()
        for frames in xml_root:
            for frame in frames:
                for annotation in frame:
                    label = annotation.findtext('type')
                    if not label:
                        raise ValueError('annotation contains detection without label')

                    labels_set.add(label)

        return labels_set
