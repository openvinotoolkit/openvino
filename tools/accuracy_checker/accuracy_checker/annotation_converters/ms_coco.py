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
import numpy as np

from ..config import BoolField
from ..utils import read_json, convert_bboxes_xywh_to_x1y1x2y2
from ..representation import DetectionAnnotation, PoseEstimationAnnotation
from .format_converter import BaseFormatConverter, FileBasedAnnotationConverter, FileBasedAnnotationConverterConfig


def get_image_annotation(image_id, annotations_):
    return list(filter(lambda x: x['image_id'] == image_id, annotations_))


def get_label_map(full_annotation, use_full_label_map=False, has_background=False):
    labels = full_annotation['categories']

    if not use_full_label_map:
        label_offset = 1 if has_background else 0
        label_id_to_label = {label['id']: label_id + label_offset for label_id, label in enumerate(labels)}
        label_map = {label_id + label_offset: label['name'] for label_id, label in enumerate(labels)}
    else:
        label_id_to_label = {label['id']: label['id'] for label in labels}
        label_map = {label['id']: label['name'] for label in labels}

    return label_map, label_id_to_label


class MSCocoDetectionConverterConfig(FileBasedAnnotationConverterConfig):
    has_background = BoolField(optional=True)
    use_full_label_map = BoolField(optional=True)


class MSCocoDetectionConverter(BaseFormatConverter):
    __provider__ = 'mscoco_detection'

    _config_validator_type = MSCocoDetectionConverterConfig

    def configure(self):
        self.annotation_file = self.config['annotation_file']
        self.has_background = self.config.get('has_background', False)
        self.use_full_label_map = self.config.get('use_full_label_map', False)

    def convert(self):
        detection_annotations = []
        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']
        annotations = full_annotation['annotations']

        label_map, label_id_to_label = get_label_map(full_annotation, self.use_full_label_map, self.has_background)

        meta = {}
        if self.has_background:
            label_map[0] = 'background'
            meta['background_label'] = 0

        meta.update({'label_map': label_map})

        for image in tqdm(image_info):
            identifier = image['file_name']
            image_annotation = get_image_annotation(image['id'], annotations)
            image_labels = [label_id_to_label[annotation['category_id']] for annotation in image_annotation]
            xmins = [annotation['bbox'][0] for annotation in image_annotation]
            ymins = [annotation['bbox'][1] for annotation in image_annotation]
            widths = [annotation['bbox'][2] for annotation in image_annotation]
            heights = [annotation['bbox'][3] for annotation in image_annotation]
            xmaxs = np.add(xmins, widths)
            ymaxs = np.add(ymins, heights)
            is_crowd = [annotation['iscrowd'] for annotation in image_annotation]
            detection_annotation = DetectionAnnotation(identifier, image_labels, xmins, ymins, xmaxs, ymaxs)
            detection_annotation.metadata['iscrowd'] = is_crowd
            detection_annotations.append(detection_annotation)

        return detection_annotations, meta


class MSCocoKeypointsConverter(FileBasedAnnotationConverter):
    __provider__ = 'mscoco_keypoints'

    def convert(self):
        keypoints_annotations = []

        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']
        annotations = full_annotation['annotations']
        label_map, _ = get_label_map(full_annotation, True)
        for image in image_info:
            identifier = image['file_name']
            image_annotation = get_image_annotation(image['id'], annotations)
            if not image_annotation:
                continue
            x_vals, y_vals, visibility, labels, areas, is_crowd, bboxes, difficult = [], [], [], [], [], [], [], []
            for target in image_annotation:
                if target['num_keypoints'] == 0:
                    difficult.append(len(x_vals))
                labels.append(target['category_id'])
                keypoints = target['keypoints']
                x_vals.append(keypoints[::3])
                y_vals.append(keypoints[1::3])
                visibility.append(keypoints[2::3])
                areas.append(target['area'])
                bboxes.append(convert_bboxes_xywh_to_x1y1x2y2(*target['bbox']))
                is_crowd.append(target['iscrowd'])
            keypoints_annotation = PoseEstimationAnnotation(
                identifier, np.array(x_vals), np.array(y_vals), np.array(visibility), np.array(labels)
            )
            keypoints_annotation.metadata['areas'] = areas
            keypoints_annotation.metadata['rects'] = bboxes
            keypoints_annotation.metadata['iscrowd'] = is_crowd
            keypoints_annotation.metadata['difficult_boxes'] = difficult

            keypoints_annotations.append(keypoints_annotation)

        return keypoints_annotations, {'label_map': label_map}
