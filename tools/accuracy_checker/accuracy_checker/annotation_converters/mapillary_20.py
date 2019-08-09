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
from ..config import PathField
from ..representation import SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..utils import get_path
from .format_converter import BaseFormatConverter, BaseFormatConverterConfig


class Mapillary20ConverterConfig(BaseFormatConverterConfig):
    data_dir = PathField(is_directory=True, optional=True)
    images_dir = PathField(optional=True, is_directory=True)
    mask_dir = PathField(optional=True, is_directory=True)


class Mapillary20Converters(BaseFormatConverter):
    __provider__ = 'mapillary_20'

    label_map = {
        0: 'Road',
        1: 'Sidewalk',
        2: 'Building',
        3: 'Wall',
        4: 'Fence',
        5: 'Pole',
        6: 'Traffic Light',
        7: 'Traffic Sign',
        8: 'Vegetation',
        9: 'Terrain',
        10: 'Sky',
        11: 'Person',
        12: 'Rider',
        13: 'Car',
        14: 'Truck',
        15: 'Bus',
        16: 'Train',
        17: 'Motorcycle',
        18: 'Bicycle',
        19: 'Ego-Vehicle'
    }

    _config_validator_type = Mapillary20ConverterConfig

    def configure(self):
        data_dir = self.config.get('data_dir')
        image_folder = self.config.get('images_dir', 'imgs')
        mask_folder = self.config.get('mask_dir', 'masks')
        if data_dir:
            image_folder = data_dir / image_folder
            mask_folder = data_dir / mask_folder
        self.images_dir = get_path(image_folder, is_directory=True)
        self.mask_dir = get_path(mask_folder, is_directory=True)

    def convert(self):
        annotations = []
        for file_in_dir in self.images_dir.iterdir():
            annotation = SegmentationAnnotation(
                str(Path(self.images_dir.name) / file_in_dir.name),
                str(Path(self.mask_dir.name) / file_in_dir.name),
                mask_loader=GTMaskLoader.PILLOW
            )

            annotations.append(annotation)

        return annotations, {'label_map': self.label_map}
