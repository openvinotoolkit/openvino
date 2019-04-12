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

from .base_representation import BaseRepresentation
from .classification_representation import Classification, ClassificationAnnotation, ClassificationPrediction
from .detection_representation import Detection, DetectionAnnotation, DetectionPrediction
from .reid_representation import (
    ReIdentificationAnnotation,
    ReIdentificationClassificationAnnotation,
    ReIdentificationPrediction
)
from .segmentation_representation import (
    SegmentationRepresentation,
    SegmentationAnnotation,
    SegmentationPrediction,
    BrainTumorSegmentationAnnotation,
    BrainTumorSegmentationPrediction
)
from .character_recognition_representation import (
    CharacterRecognition,
    CharacterRecognitionAnnotation,
    CharacterRecognitionPrediction
)
from .representaton_container import ContainerRepresentation, ContainerAnnotation, ContainerPrediction
from .regression_representation import (
    RegressionAnnotation,
    RegressionPrediction,
    FacialLandmarksAnnotation,
    FacialLandmarksPrediction,
    GazeVectorAnnotation,
    GazeVectorPrediction
)
from .multilabel_recognition import MultiLabelRecognitionAnnotation, MultiLabelRecognitionPrediction
from .super_resolution_representation import SuperResolutionAnnotation, SuperResolutionPrediction
from .text_detection_representation import TextDetectionAnnotation, TextDetectionPrediction
from .pose_estimation_representation import PoseEstimationAnnotation, PoseEstimationPrediction
from .hit_ratio_representation import HitRatio, HitRatioAnnotation, HitRatioPrediction

__all__ = [
    'BaseRepresentation',

    'Classification',
    'ClassificationAnnotation',
    'ClassificationPrediction',

    'Detection',
    'DetectionAnnotation',
    'DetectionPrediction',

    'ReIdentificationAnnotation',
    'ReIdentificationClassificationAnnotation',
    'ReIdentificationPrediction',

    'SegmentationRepresentation',
    'SegmentationAnnotation',
    'SegmentationPrediction',
    'BrainTumorSegmentationAnnotation',
    'BrainTumorSegmentationPrediction',

    'CharacterRecognition',
    'CharacterRecognitionAnnotation',
    'CharacterRecognitionPrediction',

    'ContainerRepresentation',
    'ContainerAnnotation',
    'ContainerPrediction',

    'RegressionAnnotation',
    'RegressionPrediction',
    'FacialLandmarksAnnotation',
    'FacialLandmarksPrediction',
    'GazeVectorAnnotation',
    'GazeVectorPrediction',

    'MultiLabelRecognitionAnnotation',
    'MultiLabelRecognitionPrediction',

    'SuperResolutionAnnotation',
    'SuperResolutionPrediction',

    'TextDetectionAnnotation',
    'TextDetectionPrediction',

    'PoseEstimationAnnotation',
    'PoseEstimationPrediction',

    'HitRatio',
    'HitRatioAnnotation',
    'HitRatioPrediction'
]
