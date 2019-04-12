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

from ..representation import ClassificationAnnotation, ClassificationPrediction
from ..config import NumberField, StringField
from .metric import BaseMetricConfig, PerImageEvaluationMetric
from .average_meter import AverageMeter


class ClassificationAccuracy(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy metric of classification models.
    """

    __provider__ = 'accuracy'

    annotation_types = (ClassificationAnnotation, )
    prediction_types = (ClassificationPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def loss(annotation_label, prediction_top_k_labels):
            return int(annotation_label in prediction_top_k_labels)
        self.accuracy = AverageMeter(loss)

    def validate_config(self):
        class _AccuracyValidator(BaseMetricConfig):
            top_k = NumberField(floats=False, min_value=1, optional=True)

        accuracy_validator = _AccuracyValidator(
            'accuracy',
            on_extra_argument=_AccuracyValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        accuracy_validator.validate(self.config)

    def configure(self):
        self.top_k = self.config.get('top_k', 1)

    def update(self, annotation, prediction):
        self.accuracy.update(annotation.label, prediction.top_k(self.top_k))

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()


class ClassificationAccuracyClasses(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy for each class of classification models.
    """

    __provider__ = 'accuracy_per_class'

    annotation_types = (ClassificationAnnotation, )
    prediction_types = (ClassificationPrediction, )

    def validate_config(self):
        class _AccuracyValidator(BaseMetricConfig):
            top_k = NumberField(floats=False, min_value=1, optional=True)
            label_map = StringField(optional=True)

        accuracy_validator = _AccuracyValidator(
            'accuracy',
            on_extra_argument=_AccuracyValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        accuracy_validator.validate(self.config)

    def configure(self):
        self.top_k = self.config.get('top_k', 1)
        label_map = self.config.get('label_map', 'label_map')
        self.labels = self.dataset.metadata.get(label_map)
        self.meta['names'] = list(self.labels.values())

        def loss(annotation_label, prediction_top_k_labels):
            result = np.zeros_like(list(self.labels.keys()))
            if annotation_label in prediction_top_k_labels:
                result[annotation_label] = 1

            return result

        def counter(annotation_label):
            result = np.zeros_like(list(self.labels.keys()))
            result[annotation_label] = 1
            return result

        self.accuracy = AverageMeter(loss, counter)

    def update(self, annotation, prediction):
        self.accuracy.update(annotation.label, prediction.top_k(self.top_k))

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()
