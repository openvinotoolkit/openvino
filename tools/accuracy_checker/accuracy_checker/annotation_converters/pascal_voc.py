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

from tqdm import tqdm
from pathlib import Path

from ..config import PathField, BoolField
from ..representation import DetectionAnnotation, SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..utils import get_path, read_txt, read_xml
from .format_converter import BaseFormatConverter, BaseFormatConverterConfig

_VOC_CLASSES_DETECTION = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

_VOC_CLASSES_SEGMENTATION = tuple(['__background__']) + _VOC_CLASSES_DETECTION
_SEGMENTATION_COLORS = ((
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
))


def prepare_detection_labels(has_background=True):
    num_classes = len(_VOC_CLASSES_DETECTION)
    labels_shift = 1 if has_background else 0
    reversed_label_map = dict(zip(_VOC_CLASSES_DETECTION, list(range(labels_shift, num_classes + labels_shift))))
    if has_background:
        reversed_label_map['__background__'] = 0

    return reversed_label_map


def reverse_label_map(label_map):
    return {value: key for key, value in label_map.items()}


class PascalVOCSegmentationConverterConfig(BaseFormatConverterConfig):
    image_set_file = PathField()
    images_dir = PathField(optional=True, is_directory=True)
    mask_dir = PathField(optional=True, is_directory=True)


class PascalVOCSegmentationConverter(BaseFormatConverter):
    __provider__ = 'voc_segmentation'

    _config_validator_type = PascalVOCSegmentationConverterConfig

    def configure(self):
        self.image_set_file = self.config['image_set_file']
        self.image_dir = self.config.get('images_dir')
        if not self.image_dir:
            self.image_dir = get_path(self.image_set_file.parents[-2] / 'JPEGImages', is_directory=True)

        self.mask_dir = self.config.get('mask_dir')
        if not self.mask_dir:
            self.mask_dir = get_path(self.image_set_file.parents[-2] / 'SegmentationClass', is_directory=True)

    def convert(self):

        annotations = []
        for image in read_txt(self.image_set_file):
            annotation = SegmentationAnnotation(
                str(Path(self.image_dir.name) / '{}.jpg'.format(image)),
                str(Path(self.mask_dir.name) / '{}.png'.format(image)),
                mask_loader=GTMaskLoader.SCIPY
            )

            annotations.append(annotation)

        meta = {
            'label_map': dict(enumerate(_VOC_CLASSES_SEGMENTATION)),
            'background_label': 0,
            'segmentation_colors': _SEGMENTATION_COLORS
        }

        return annotations, meta


class PascalVOCDetectionConverterConfig(BaseFormatConverterConfig):
    image_set_file = PathField()
    annotations_dir = PathField(is_directory=True)
    images_dir = PathField(optional=True, is_directory=True)
    has_background = BoolField(optional=True)


class PascalVOCDetectionConverter(BaseFormatConverter):
    __provider__ = 'voc07'

    _config_validator_type = PascalVOCDetectionConverterConfig

    def configure(self):
        self.image_set_file = self.config['image_set_file']
        self.image_dir = self.config.get('images_dir')
        if not self.image_dir:
            self.image_dir = get_path(self.image_set_file.parents[-2] / 'JPEGImages')
        self.annotations_dir = self.config['annotations_dir']
        self.has_background = self.config.get('has_background', True)

    def convert(self):
        class_to_ind = prepare_detection_labels(self.has_background)

        detections = []
        for image in tqdm(read_txt(self.image_set_file, sep=None)):
            root = read_xml(self.annotations_dir / '{}.xml'.format(image))

            identifier = root.find('.//filename').text
            get_path(self.image_dir / identifier)

            labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
            difficult_indices = []
            for entry in root:
                if not entry.tag.startswith('object'):
                    continue

                bbox = entry.find('bndbox')
                difficult = int(entry.find('difficult').text)

                if difficult == 1:
                    difficult_indices.append(len(labels))

                labels.append(class_to_ind[entry.find('name').text])
                x_mins.append(float(bbox.find('xmin').text) - 1)
                y_mins.append(float(bbox.find('ymin').text) - 1)
                x_maxs.append(float(bbox.find('xmax').text) - 1)
                y_maxs.append(float(bbox.find('ymax').text) - 1)

            image_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
            image_annotation.metadata['difficult_boxes'] = difficult_indices

            detections.append(image_annotation)

        meta = {'label_map': reverse_label_map(class_to_ind)}
        if self.has_background:
            meta['background_label'] = 0

        return detections, meta
