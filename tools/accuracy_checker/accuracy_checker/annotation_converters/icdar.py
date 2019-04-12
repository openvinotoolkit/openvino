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
from ..representation import TextDetectionAnnotation, CharacterRecognitionAnnotation
from ..utils import read_txt
from .format_converter import  FileBasedAnnotationConverter, DirectoryBasedAnnotationConverter


class ICDAR15DetectionDatasetConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'icdar15_detection'

    def convert(self):
        annotations = []

        for gt_file in self.data_dir.iterdir():
            gt_file_name = str(gt_file.parts[-1])
            identifier = '{}.jpg'.format(gt_file_name.split('gt_')[-1].split('.txt')[0])
            all_points, transcriptions, difficult = [], [], []

            for text_area in read_txt(gt_file):
                text_annotation = text_area.split(',')
                transcription = text_annotation[-1]
                points = np.reshape(list(map(float, text_annotation[:8])), (-1, 2))
                if transcription == '###':
                    difficult.append(len(transcriptions))
                all_points.append(points)
                transcriptions.append(transcription)
            annotation = TextDetectionAnnotation(identifier, all_points, transcriptions)
            annotation.metadata['difficult_boxes'] = difficult
            annotations.append(annotation)

        return annotations, None


class ICDAR13RecognitionDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'icdar13_recognition'

    supported_symbols = '0123456789abcdefghijklmnopqrstuvwxyz'

    def convert(self):
        annotations = []

        for line in read_txt(self.annotation_file):
            identifier, text = line.strip().split(' ')
            annotations.append(CharacterRecognitionAnnotation(identifier, text))

        label_map = {ind: str(key) for ind, key in enumerate(self.supported_symbols)}

        return annotations, {'label_map': label_map, 'blank_label': len(label_map)}
