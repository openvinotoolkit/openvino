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

from .adapter import Adapter, AdapterField

from .action_recognition import ActionDetection
from .text_detection import TextDetectionAdapter, LPRAdapter, BeamSearchDecoder
from .image_processing import SuperResolutionAdapter
from .attributes_recognition import (
    HeadPoseEstimatorAdapter,
    VehicleAttributesRecognitionAdapter,
    PersonAttributesAdapter,
    AgeGenderAdapter,
    LandmarksRegressionAdapter,
    GazeEstimationAdapter
)

from .reidentification import ReidAdapter
from .detection import TinyYOLOv1Adapter, SSDAdapter, FacePersonAdapter, YoloV2Adapter, YoloV3Adapter
from .classification import ClassificationAdapter
from .segmentation import SegmentationAdapter, BrainTumorSegmentationAdapter
from .pose_estimation import HumanPoseAdapter

from .dummy_adapters import XML2DetectionAdapter

from .hit_ratio import HitRatioAdapter

__all__ = [
    'Adapter',
    'AdapterField',

    'XML2DetectionAdapter',

    'ClassificationAdapter',

    'SSDAdapter',
    'TinyYOLOv1Adapter',
    'YoloV2Adapter',
    'YoloV3Adapter',
    'FacePersonAdapter',

    'SegmentationAdapter',
    'BrainTumorSegmentationAdapter',

    'ReidAdapter',

    'SuperResolutionAdapter',

    'HeadPoseEstimatorAdapter',
    'VehicleAttributesRecognitionAdapter',
    'PersonAttributesAdapter',
    'AgeGenderAdapter',
    'LandmarksRegressionAdapter',
    'GazeEstimationAdapter',

    'TextDetectionAdapter',

    'BeamSearchDecoder',
    'LPRAdapter',

    'HumanPoseAdapter',

    'ActionDetection',

    'HitRatioAdapter'
]
