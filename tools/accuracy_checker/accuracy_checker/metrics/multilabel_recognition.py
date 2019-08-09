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
from .metric import PerImageEvaluationMetric, BaseMetricConfig
from ..representation import MultiLabelRecognitionAnnotation, MultiLabelRecognitionPrediction
from ..config import StringField, BoolField


class MultiLabelConfigValidator(BaseMetricConfig):
    label_map = StringField(optional=True)
    calculate_average = BoolField(optional=True)


class MultiLabelMetric(PerImageEvaluationMetric):
    annotation_types = (MultiLabelRecognitionAnnotation,)
    prediction_types = (MultiLabelRecognitionPrediction,)
    _config_validator_type = MultiLabelConfigValidator

    def configure(self):
        label_map = self.config.get('label_map', 'label_map')
        self.labels = self.dataset.metadata.get(label_map)
        self.calculate_average = self.config.get('calculate_average', True)

        self.meta['scale'] = 1
        self.meta['postfix'] = ''
        self.meta['calculate_mean'] = False
        self.meta['names'] = list(self.labels.values())
        if self.calculate_average:
            self.meta['names'].append('average')
        self.tp = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.fp = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.tn = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.fn = np.zeros_like(list(self.labels.keys()), dtype=np.float)

        self.counter = np.zeros_like(list(self.labels.keys()), dtype=np.float)

    def update(self, annotation, prediction):
        def loss(annotation_labels, prediction_labels):
            tp_result = np.zeros_like(list(self.labels.keys()), dtype=np.float)
            fp_results = np.zeros_like(list(self.labels.keys()), dtype=np.float)
            tn_results = np.zeros_like(list(self.labels.keys()), dtype=np.float)
            fn_results = np.zeros_like(list(self.labels.keys()), dtype=np.float)

            for index, label in enumerate(annotation_labels):
                if label == 1 and label == prediction_labels[index]:
                    tp_result[index] = 1.
                    continue

                if label == 1 and label != prediction_labels[index]:
                    fn_results[index] = 1.
                    continue

                if label == 0 and label == prediction_labels[index]:
                    tn_results[index] = 1.
                    continue

                if label == 0 and label != prediction_labels[index]:
                    fp_results[index] = 1.
                    continue

            return tp_result, fp_results, tn_results, fn_results

        def counter(annotation_label):
            count = np.zeros_like(annotation_label, dtype=float)
            cond = np.where(np.array(annotation_label) != -1)
            count[cond] = 1.
            return count

        tp_upd, fp_upd, tn_upd, fn_upd = loss(annotation.multi_label, prediction.multi_label)
        self.tp = np.add(self.tp, tp_upd)
        self.fp = np.add(self.fp, fp_upd)
        self.tn = np.add(self.tn, tn_upd)
        self.fn = np.add(self.fn, fn_upd)

        self.counter = np.add(self.counter, counter(annotation.multi_label))

    def evaluate(self, annotations, predictions):
        pass


class MultiLabelAccuracy(MultiLabelMetric):
    __provider__ = 'multi_accuracy'

    def evaluate(self, annotations, predictions):
        tp_tn = np.add(self.tp, self.tn, dtype=float)
        per_class = np.divide(tp_tn, self.counter, out=np.zeros_like(tp_tn, dtype=float), where=self.counter != 0)
        average = np.sum(tp_tn) / np.sum(self.counter)

        return [*per_class, average]


class MultiLabelPrecision(MultiLabelMetric):
    __provider__ = 'multi_precision'

    def evaluate(self, annotations, predictions):
        tp_fp = np.add(self.tp, self.fp, dtype=float)
        per_class = np.divide(self.tp, tp_fp, out=np.zeros_like(self.tp, dtype=float), where=tp_fp != 0)
        if not self.calculate_average:
            return per_class
        average = np.sum(self.tp) / np.sum(tp_fp)

        return [*per_class, average]


class MultiLabelRecall(MultiLabelMetric):
    __provider__ = 'multi_recall'

    def evaluate(self, annotations, predictions):
        tp_fn = np.add(self.tp, self.fn, dtype=float)
        per_class = np.divide(self.tp, tp_fn, out=np.zeros_like(self.tp, dtype=float), where=tp_fn != 0)
        if not self.calculate_average:
            return per_class
        average = np.sum(self.tp) / np.sum(tp_fn)

        return [*per_class, average]


class F1Score(PerImageEvaluationMetric):
    __provider__ = 'f1-score'
    annotation_types = (MultiLabelRecognitionAnnotation,)
    prediction_types = (MultiLabelRecognitionPrediction,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = MultiLabelPrecision(self.config, self.dataset)
        self.recall = MultiLabelRecall(self.config, self.dataset)

    def validate_config(self):
        class _F1ScoreValidator(BaseMetricConfig):
            label_map = StringField(optional=True)
            calculate_average = BoolField(optional=True)

        f1_score_config_validator = _F1ScoreValidator(
            'f1_score', on_extra_argument=_F1ScoreValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        f1_score_config_validator.validate(self.config)

    def configure(self):
        label_map = self.config.get('label_map', 'label_map')
        self.labels = self.dataset.metadata.get(label_map)
        self.calculate_average = self.config.get('calculate_average', True)
        self.meta['names'] = list(self.labels.values())
        if self.calculate_average:
            self.meta['names'].append('average')

        self.meta['scale'] = 1
        self.meta['postfix'] = ''
        self.meta['calculate_mean'] = False
        self.meta['names'] = list(self.labels.values()) + ['average']

    def update(self, annotation, prediction):
        self.precision.update(annotation, prediction)
        self.recall.update(annotation, prediction)

    def evaluate(self, annotations, predictions):
        precisions = self.precision.evaluate(annotations, predictions)
        recalls = self.recall.evaluate(annotations, predictions)

        precision_add = np.add(precisions[:-1], recalls[:-1], dtype=float)
        precision_multiply = np.multiply(precisions[:-1], recalls[:-1], dtype=float)

        per_class = 2 * np.divide(
            precision_multiply, precision_add, out=np.zeros_like(precision_multiply, dtype=float),
            where=precision_add != 0
        )
        if not self.calculate_average:
            return per_class

        average = 2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1])

        return [*per_class, average]
