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

from collections import defaultdict
from pathlib import Path

from ..config import PathField
from ..representation import ReIdentificationClassificationAnnotation
from ..utils import read_txt

from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class FaceReidPairwiseConverterConfig(BaseFormatConverterConfig):
    pairs_file = PathField()
    train_file = PathField(optional=True)
    landmarks_file = PathField(optional=True)


class FaceReidPairwiseConverter(BaseFormatConverter):
    __provider__ = 'face_reid_pairwise'

    _config_validator_type = FaceReidPairwiseConverterConfig

    def configure(self):
        self.pairs_file = self.config['pairs_file']
        self.train_file = self.config.get('train_file')
        self.landmarks_file = self.config.get('landmarks_file')

    def convert(self):
        landmarks_map = {}
        if self.landmarks_file:
            for landmark_line in read_txt(self.landmarks_file):
                landmark_line = landmark_line.split('\t')
                landmarks_map[landmark_line[0]] = [int(point) for point in landmark_line[1:]]

        test_annotations = self.prepare_annotation(self.pairs_file, True, landmarks_map)
        if self.train_file:
            train_annotations = self.prepare_annotation(self.train_file, True, landmarks_map)
            test_annotations += train_annotations

        return test_annotations, None

    @staticmethod
    def get_image_name(person, image_id):
        image_path_pattern = '{}/{}_{}{}.jpg'
        return image_path_pattern.format(person, person, '0' * (4 - len(image_id)), image_id)

    def convert_positive(self, pairs, all_images):
        positives = defaultdict(set)
        for data in pairs:
            image1 = self.get_image_name(data[0], data[1])
            image2 = self.get_image_name(data[0], data[2])
            positives[image1].add(image2)
            all_images.add(image1)
            all_images.add(image2)

        return positives, all_images

    def convert_negative(self, pairs, all_images):
        negatives = defaultdict(set)
        for data in pairs:
            image1 = self.get_image_name(data[0], data[1])
            image2 = self.get_image_name(data[2], data[3])
            negatives[image1].add(image2)
            all_images.add(image1)
            all_images.add(image2)

        return negatives, all_images

    def prepare_annotation(self, ann_file: Path, train=False, landmarks_map=None):
        positive_pairs, negative_pairs = [], []
        ann_lines = read_txt(ann_file)
        for line in ann_lines[1:]:  # skip header
            pair = line.strip().split()
            if len(pair) == 3:
                positive_pairs.append(pair)
            elif len(pair) == 4:
                negative_pairs.append(pair)

        all_images = set()
        positive_data, all_images = self.convert_positive(positive_pairs, all_images)
        negative_data, all_images = self.convert_negative(negative_pairs, all_images)

        annotations = []
        for image in all_images:
            annotation = ReIdentificationClassificationAnnotation(image, positive_data[image], negative_data[image])

            if landmarks_map:
                image_landmarks = landmarks_map.get(image)
                annotation.metadata['keypoints'] = image_landmarks

            if train:
                annotation.metadata['train'] = True

            annotations.append(annotation)

        return annotations
