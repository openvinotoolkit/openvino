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

import pytest
import numpy as np
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.presenters import EvaluationResult
from .common import single_class_dataset, multi_class_dataset, make_segmentation_representation


def create_config(metric_name, use_argmax=False):
    return {'annotation': 'mocked', 'metrics': [{'type': metric_name, 'use_argmax': use_argmax}]}


def generate_expected_result(values, metric_name, labels=None):
    meta = {'names': list(labels.values())} if labels else {}

    return EvaluationResult(pytest.approx(values), None, metric_name, None, meta)


class TestPixelAccuracy:
    name = 'segmentation_accuracy'

    def test_one_class(self):
        annotations = make_segmentation_representation(np.array([[0, 0], [0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0], [0, 0]]), False)
        dispatcher = MetricsExecutor(create_config(self.name), single_class_dataset())
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result(1.0, self.name)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class_not_matched(self):
        annotations = make_segmentation_representation(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), False)
        dispatcher = MetricsExecutor(create_config(self.name), multi_class_dataset())
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result(0.0, self.name)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class(self):
        annotations = make_segmentation_representation(np.array([[1, 0, 3, 0, 0], [0, 0, 0, 0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[1, 2, 3, 2, 3], [0, 0, 0, 0, 0]]), False)
        dispatcher = MetricsExecutor(create_config(self.name), multi_class_dataset())
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result((5.0+1.0+1.0)/(8.0+1.0+1.0), self.name)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected


class TestMeanAccuracy:
    name = 'mean_accuracy'

    def test_one_class(self):
        annotations = make_segmentation_representation(np.array([[0, 0], [0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0], [0, 0]]), False)
        dataset = single_class_dataset()
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result([1.0, 0.0], self.name, dataset.labels)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class_not_matched(self):
        annotations = make_segmentation_representation(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), False)
        dataset = multi_class_dataset()
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result([0.0, 0.0, 0.0, 0.0], self.name, dataset.labels)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class(self):
        dataset = multi_class_dataset()
        annotations = make_segmentation_representation(np.array([[1, 2, 3, 2, 3], [0, 0, 0, 0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[1, 0, 3, 0, 0], [0, 0, 0, 0, 0]]), False)
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result([1.0, 1.0, 0.0, 0.5], self.name, dataset.labels)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected


class TestMeanIOU:
    name = 'mean_iou'

    def test_one_class(self):
        annotations = make_segmentation_representation(np.array([[0, 0], [0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0], [0, 0]]), False)
        dataset = single_class_dataset()
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result([1.0, 0.0], self.name, dataset.labels)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class_not_matched(self):
        annotations = make_segmentation_representation(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), False)
        dataset = multi_class_dataset()
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result([0.0, 0.0, 0.0, 0.0], self.name, dataset.labels)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class(self):
        dataset = multi_class_dataset()
        annotations = make_segmentation_representation(np.array([[1, 2, 3, 2, 3], [0, 0, 0, 0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[1, 0, 3, 0, 0], [0, 0, 0, 0, 0]]), False)
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result([0.625, 1.0, 0.0, 0.5], self.name, dataset.labels)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected


class TestSegmentationFWAcc:
    name = 'frequency_weighted_accuracy'

    def test_one_class(self):
        annotations = make_segmentation_representation(np.array([[0, 0], [0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0], [0, 0]]), False)
        dataset = single_class_dataset()
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result(1.0, self.name)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class_not_matched(self):
        annotations = make_segmentation_representation(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), True)
        predictions = make_segmentation_representation(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), False)
        dataset = multi_class_dataset()
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result(0.0, self.name)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_multi_class(self):
        dataset = multi_class_dataset()
        annotations = make_segmentation_representation(np.array([[1, 2, 3, 2, 3], [0, 0, 0, 0, 0]]), True)
        predictions = make_segmentation_representation(np.array([[1, 0, 3, 0, 0], [0, 0, 0, 0, 0]]), False)
        dispatcher = MetricsExecutor(create_config(self.name), dataset)
        dispatcher.update_metrics_on_batch(annotations, predictions)
        expected = generate_expected_result(0.5125, self.name)
        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected
