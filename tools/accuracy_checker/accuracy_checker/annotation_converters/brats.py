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

from ..representation import BrainTumorSegmentationAnnotation
from ..utils import get_path
from ..config import StringField
from .format_converter import BaseFormatConverter, DirectoryBasedAnnotationConverterConfig


class BratsConverterConfig(DirectoryBasedAnnotationConverterConfig):
    image_folder = StringField(optional=True)
    mask_folder = StringField(optional=True)


class BratsConverter(BaseFormatConverter):
    __provider__ = 'brats'

    _config_validator_type = BratsConverterConfig

    def configure(self):
        self.data_dir = self.config['data_dir']
        self.image_folder = self.config.get('image_folder', 'imagesTr')
        self.mask_folder = self.config.get('mask_folder', 'labelsTr')

    def convert(self):
        mask_folder = Path(self.mask_folder)
        image_folder = Path(self.image_folder)
        image_dir = get_path(self.data_dir / image_folder, is_directory=True)

        annotations = []
        for file_in_dir in image_dir.iterdir():
            annotation = BrainTumorSegmentationAnnotation(
                str(image_folder / file_in_dir.parts[-1]),
                str(mask_folder / file_in_dir.parts[-1]),
            )

            annotations.append(annotation)

        return annotations, None
