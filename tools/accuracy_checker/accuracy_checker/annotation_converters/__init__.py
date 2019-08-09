"""
Copyright (c) 2018 Intel Corporation

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
from .format_converter import BaseFormatConverter
from .convert import make_subset, save_annotation
from .market1501 import Market1501Converter
from .mars import MARSConverter
from .pascal_voc import PascalVOCDetectionConverter
from .sample_converter import SampleConverter
from .wider import WiderFormatConverter
from .detection_opencv_storage import DetectionOpenCVStorageFormatConverter
from .bitvehicle import BITVehicle, BITVehicleJSON
from .lfw import FaceReidPairwiseConverter
from .vgg_face_regression import LandmarksRegression
from .mighty import MightyFormatConverter
from .super_resolution_converter import SRConverter
from .mapillary_20 import Mapillary20Converters
from .imagenet import ImageNetFormatConverter
from .icdar import ICDAR13RecognitionDatasetConverter, ICDAR15DetectionDatasetConverter
from .ms_coco import MSCocoDetectionConverter, MSCocoKeypointsConverter
from .cityscapes import CityscapesConverter
from .ncf_converter import NCFConverter
from .brats import BratsConverter

__all__ = [
    'BaseFormatConverter',
    'make_subset',
    'save_annotation',

    'ImageNetFormatConverter',
    'Market1501Converter',
    'SampleConverter',
    'PascalVOCDetectionConverter',
    'WiderFormatConverter',
    'MARSConverter',
    'DetectionOpenCVStorageFormatConverter',
    'BITVehicle',
    'BITVehicleJSON',
    'FaceReidPairwiseConverter',
    'LandmarksRegression',
    'MightyFormatConverter',
    'SRConverter',
    'Mapillary20Converters',
    'ICDAR13RecognitionDatasetConverter',
    'ICDAR15DetectionDatasetConverter',
    'MSCocoKeypointsConverter',
    'MSCocoDetectionConverter',
    'CityscapesConverter',
    'NCFConverter',
    'BratsConverter',
]
