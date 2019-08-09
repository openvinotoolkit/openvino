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

from ..representation import SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..utils import read_txt
from .format_converter import FileBasedAnnotationConverter


class MightyFormatConverter(FileBasedAnnotationConverter):
    __provider__ = 'mighty'

    label_map = {0: 'BG', 1: 'road', 2: 'curbs', 3: 'marks'}

    def convert(self):
        annotations = []
        for line in read_txt(self.annotation_file):
            identifier, mask = line.split()
            annotations.append(SegmentationAnnotation(identifier, mask, mask_loader=GTMaskLoader.PILLOW))

        return annotations, {'label_map': self.label_map}
