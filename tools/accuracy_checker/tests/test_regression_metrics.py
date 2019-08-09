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
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.representation import RegressionPrediction, RegressionAnnotation
from accuracy_checker.presenters import EvaluationResult


class TestRegressionMetric:
    def setup_method(self):
        self.module = 'accuracy_checker.metrics.metric_evaluator'

    def test_mae_with_zero_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3)]
        predictions = [RegressionPrediction('identifier', 3)]
        config = [{'type': 'mae'}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0]),
            None,
            'mae',
            'mae',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_with_negative_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'mae'}]
        expected = EvaluationResult(
            pytest.approx([3.0, 1.0]),
            None,
            'mae',
            'mae',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_with_positive_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier2', -3)]
        config = [{'type': 'mae'}]
        expected = EvaluationResult(
            pytest.approx([3.0, 1.0]),
            None,
            'mae',
            'mae',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mse_with_zero_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3)]
        predictions = [RegressionPrediction('identifier', 3)]
        config = [{'type': 'mse'}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0]),
            None,
            'mse',
            'mse',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mse_with_negative_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'mse'}]
        expected = EvaluationResult(
            pytest.approx([10.0, 6.0]),
            None,
            'mse',
            'mse',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mse_with_positive_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier2', -3)]
        config = [{'type': 'mse'}]
        expected = EvaluationResult(
            pytest.approx([10.0, 6.0]),
            None,
            'mse',
            'mse',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_missed_interval(self):
        config = [{'type': 'mae_on_interval'}]
        with pytest.raises(ValueError):
            MetricsExecutor(config, None)

    def test_mae_on_interval_default_all_missed(self):
        annotations = [RegressionAnnotation('identifier', -2)]
        predictions = [RegressionPrediction('identifier', 1)]
        config = [{'type': 'mae_on_interval', 'end': 1}]
        expected = EvaluationResult(
            pytest.approx([0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {'postfix': ' ', 'scale': 1, 'names': [], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        with pytest.warns(UserWarning) as warnings:
            for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
                assert len(warnings) == 1
                assert evaluation_result == expected

    def test_mae_on_interval_default_all_not_in_range_not_ignore_out_of_range(self):
        annotations = [RegressionAnnotation('identifier', -1), RegressionAnnotation('identifier', 2)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier', 2)]
        expected = EvaluationResult(
            pytest.approx([2.0, 0.0, 0.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': ['mean: < 0.0', 'std: < 0.0', 'mean: > 1.0', 'std: > 1.0'],
                'calculate_mean': False
            }
        )
        config = [{'type': 'mae_on_interval', 'end': 1, 'ignore_values_not_in_interval': False}]
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_values_in_range(self):
        annotations = [RegressionAnnotation('identifier', 0.5), RegressionAnnotation('identifier', 0.5)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier', 0.25)]
        config = [{'type': 'mae_on_interval', 'end': 1}]
        expected = EvaluationResult(
            pytest.approx([0.375, 0.125]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean: <= 0.0 < 1.0', 'std: <= 0.0 < 1.0'], 'calculate_mean': False}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_default_not_ignore_out_of_range(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier', 0.5)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 2),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'end': 1, 'ignore_values_not_in_interval': False}]
        expected = EvaluationResult(
            pytest.approx([2.0, 0.0, 0.5, 0.0,  0.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': [
                    'mean: < 0.0',
                    'std: < 0.0',
                    'mean: <= 0.0 < 1.0',
                    'std: <= 0.0 < 1.0',
                    'mean: > 1.0',
                    'std: > 1.0'
                ],
                'calculate_mean': False
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_with_given_interval(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier',  1)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 3),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'intervals': [0.0, 2.0, 4.0]}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0, 1.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': ['mean: <= 0.0 < 2.0', 'std: <= 0.0 < 2.0', 'mean: <= 2.0 < 4.0', 'std: <= 2.0 < 4.0'],
                'calculate_mean': False
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_with_repeated_values(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier', 1)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 3),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'intervals': [0.0, 2.0, 2.0, 4.0]}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0, 1.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': ['mean: <= 0.0 < 2.0', 'std: <= 0.0 < 2.0', 'mean: <= 2.0 < 4.0', 'std: <= 2.0 < 4.0'],
                'calculate_mean': False
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_with_unsorted_values(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier',  1)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 3),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'intervals': [2.0,  0.0, 4.0]}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0, 1.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            {
                'postfix': ' ', 'scale': 1,
                'names': ['mean: <= 0.0 < 2.0', 'std: <= 0.0 < 2.0', 'mean: <= 2.0 < 4.0', 'std: <= 2.0 < 4.0'],
                'calculate_mean': False
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected
