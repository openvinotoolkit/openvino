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
from accuracy_checker.config import ConfigError
from accuracy_checker.metrics import ClassificationAccuracy, MetricsExecutor
from accuracy_checker.metrics.metric import Metric
from accuracy_checker.representation import (
    ClassificationAnnotation,
    ClassificationPrediction,
    ContainerAnnotation,
    ContainerPrediction,
    DetectionAnnotation,
    DetectionPrediction
)
from .common import DummyDataset


class TestMetric:
    def setup_method(self):
        self.module = 'accuracy_checker.metrics.metric_evaluator'

    def test_missed_metrics_raises_config_error_exception(self):
        config = {'annotation': 'custom'}

        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_missed_metrics_raises_config_error_exception_with_custom_name(self):
        config = {'name': 'some_name', 'annotation': 'custom'}

        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_empty_metrics_raises_config_error_exception(self):
        config = {'annotation': 'custom', 'metrics': []}

        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_metrics_with_empty_entry_raises_config_error_exception(self):
        config = {'annotation': 'custom', 'metrics': [{}]}

        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_missed_metric_type_raises_config_error_exception(self):
        config = {'annotation': 'custom', 'metrics': [{'undefined': ''}]}

        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_undefined_metric_type_raises_config_error_exception(self):
        config = {'annotation': 'custom', 'metrics': [{'type': ''}]}

        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_accuracy_arguments(self):
        config = {'annotation': 'custom', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        assert len(dispatcher.metrics) == 1
        _, accuracy_metric, _, _, _ = dispatcher.metrics[0]
        assert isinstance(accuracy_metric, ClassificationAccuracy)
        assert accuracy_metric.top_k == 1

    def test_accuracy_with_several_annotation_source_raises_config_error_exception(self):
        config = {
            'annotation': 'custom',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'annotation_source': 'annotation1, annotation2'}]
        }
        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_accuracy_with_several_prediction_source_raises_value_error_exception(self):
        config = {
            'annotation': 'custom',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'prediction_source': 'prediction1, prediction2'}]
        }
        with pytest.raises(ConfigError):
            MetricsExecutor(config, None)

    def test_accuracy_on_container_with_wrong_annotation_source_name_raise_config_error_exception(self):
        annotations = [ContainerAnnotation({'annotation': ClassificationAnnotation('identifier', 3)})]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1, 'annotation_source': 'a'}]}

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_with_wrong_annotation_type_raise_config_error_exception(self):
        annotations = [DetectionAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {
            'annotation': 'mocked',
            'metrics': [{'type': 'accuracy', 'top_k': 1}]
        }

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_with_unsupported_annotations_in_container_raise_config_error_exception(self):
        annotations = [ContainerAnnotation({'annotation': DetectionAnnotation('identifier', 3)})]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {
            'annotation': 'mocked',
            'metrics': [{'type': 'accuracy', 'top_k': 1}]
        }

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_with_unsupported_annotation_type_as_annotation_source_for_container_raises_config_error(self):
        annotations = [ContainerAnnotation({'annotation': DetectionAnnotation('identifier', 3)})]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {
            'annotation': 'mocked',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'annotation_source': 'annotation'}]
        }

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_on_annotation_container_with_several_suitable_representations_config_value_error_exception(self):
        annotations = [ContainerAnnotation({
            'annotation1': ClassificationAnnotation('identifier', 3),
            'annotation2': ClassificationAnnotation('identifier', 3)
        })]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_with_wrong_prediction_type_raise_config_error_exception(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [DetectionPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_with_unsupported_prediction_in_container_raise_config_error_exception(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ContainerPrediction({'prediction': DetectionPrediction('identifier', [1.0, 1.0, 1.0, 4.0])})]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_with_unsupported_prediction_type_as_prediction_source_for_container_raises_config_error(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ContainerPrediction({'prediction': DetectionPrediction('identifier', [1.0, 1.0, 1.0, 4.0])})]
        config = {
            'annotation': 'mocked',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'prediction_source': 'prediction'}]
        }

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_accuracy_on_prediction_container_with_several_suitable_representations_raise_config_error_exception(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ContainerPrediction({
            'prediction1': ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0]),
            'prediction2': ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])
        })]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        with pytest.raises(ConfigError):
            dispatcher.update_metrics_on_batch(annotations, predictions)

    def test_complete_accuracy(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_complete_accuracy_with_container_default_sources(self):
        annotations = [ContainerAnnotation({'a': ClassificationAnnotation('identifier', 3)})]
        predictions = [ContainerPrediction({'p': ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])})]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_complete_accuracy_with_container_sources(self):
        annotations = [ContainerAnnotation({'a': ClassificationAnnotation('identifier', 3)})]
        predictions = [ContainerPrediction({'p': ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])})]
        config = {
            'annotation': 'mocked',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'annotation_source': 'a', 'prediction_source': 'p'}]
        }

        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_zero_accuracy(self):
        annotation = [ClassificationAnnotation('identifier', 2)]
        prediction = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}

        dispatcher = MetricsExecutor(config, None)

        for _, evaluation_result in dispatcher.iterate_metrics([annotation], [prediction]):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == 0.0
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_complete_accuracy_top_3(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 3.0, 4.0, 2.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 3}]}

        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_zero_accuracy_top_3(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [5.0, 3.0, 4.0, 1.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 3}]}

        dispatcher = MetricsExecutor(config, None)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == 0.0
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_reference_is_10_by_config(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [5.0, 3.0, 4.0, 1.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 3, 'reference': 10}]}

        dispatcher = MetricsExecutor(config, None)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == 0.0
            assert evaluation_result.reference_value == 10
            assert evaluation_result.threshold is None

    def test_threshold_is_10_by_config(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [5.0, 3.0, 4.0, 1.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 3, 'threshold': 10}]}

        dispatcher = MetricsExecutor(config, None)

        for _, evaluation_result in dispatcher.iterate_metrics([annotations], [predictions]):
            assert evaluation_result.name == 'accuracy'
            assert evaluation_result.evaluated_value == 0.0
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold == 10

    def test_classification_per_class_accuracy_fully_zero_prediction(self):
        annotation = ClassificationAnnotation('identifier', 0)
        prediction = ClassificationPrediction('identifier', [1.0, 2.0])
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy_per_class', 'top_k': 1}]}
        dataset = DummyDataset(label_map={0: '0', 1: '1'})
        dispatcher = MetricsExecutor(config, dataset)
        dispatcher.update_metrics_on_batch([annotation], [prediction])
        for _, evaluation_result in dispatcher.iterate_metrics([annotation], [prediction]):
            assert evaluation_result.name == 'accuracy_per_class'
            assert len(evaluation_result.evaluated_value) == 2
            assert evaluation_result.evaluated_value[0] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[1] == pytest.approx(0.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_classification_per_class_accuracy_partially_zero_prediction(self):
        annotation = [ClassificationAnnotation('identifier', 1)]
        prediction = [ClassificationPrediction('identifier', [1.0, 2.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy_per_class', 'top_k': 1}]}
        dataset = DummyDataset(label_map={0: '0', 1: '1'})
        dispatcher = MetricsExecutor(config, dataset)

        dispatcher.update_metrics_on_batch(annotation, prediction)

        for _, evaluation_result in dispatcher.iterate_metrics(annotation, prediction):
            assert evaluation_result.name == 'accuracy_per_class'
            assert len(evaluation_result.evaluated_value) == 2
            assert evaluation_result.evaluated_value[0] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[1] == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_classification_per_class_accuracy_complete_prediction(self):
        annotation = [ClassificationAnnotation('identifier_1', 1), ClassificationAnnotation('identifier_2', 0)]
        prediction = [
            ClassificationPrediction('identifier_1', [1.0, 2.0]),
            ClassificationPrediction('identifier_2', [2.0, 1.0])
        ]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy_per_class', 'top_k': 1}]}
        dataset = DummyDataset(label_map={0: '0', 1: '1'})
        dispatcher = MetricsExecutor(config, dataset)

        dispatcher.update_metrics_on_batch(annotation, prediction)

        for _, evaluation_result in dispatcher.iterate_metrics(annotation, prediction):
            assert evaluation_result.name == 'accuracy_per_class'
            assert len(evaluation_result.evaluated_value) == 2
            assert evaluation_result.evaluated_value[0] == pytest.approx(1.0)
            assert evaluation_result.evaluated_value[1] == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_classification_per_class_accuracy_partially_prediction(self):
        annotation = [
            ClassificationAnnotation('identifier_1', 1),
            ClassificationAnnotation('identifier_2', 0),
            ClassificationAnnotation('identifier_3', 0)
        ]
        prediction = [
            ClassificationPrediction('identifier_1', [1.0, 2.0]),
            ClassificationPrediction('identifier_2', [2.0, 1.0]),
            ClassificationPrediction('identifier_3', [1.0, 5.0])
        ]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy_per_class', 'top_k': 1}]}
        dataset = DummyDataset(label_map={0: '0', 1: '1'})
        dispatcher = MetricsExecutor(config, dataset)

        dispatcher.update_metrics_on_batch(annotation, prediction)

        for _, evaluation_result in dispatcher.iterate_metrics(annotation, prediction):
            assert evaluation_result.name == 'accuracy_per_class'
            assert len(evaluation_result.evaluated_value) == 2
            assert evaluation_result.evaluated_value[0] == pytest.approx(0.5)
            assert evaluation_result.evaluated_value[1] == pytest.approx(1.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_classification_per_class_accuracy_prediction_top3_zero(self):
        annotation = [ClassificationAnnotation('identifier_1', 0), ClassificationAnnotation('identifier_2', 1)]
        prediction = [
            ClassificationPrediction('identifier_1', [1.0, 2.0, 3.0, 4.0]),
            ClassificationPrediction('identifier_2', [2.0, 1.0, 3.0, 4.0])
        ]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy_per_class', 'top_k': 3}]}
        dataset = DummyDataset(label_map={0: '0', 1: '1', 2: '2', 3: '3'})
        dispatcher = MetricsExecutor(config, dataset)

        dispatcher.update_metrics_on_batch(annotation, prediction)

        for _, evaluation_result in dispatcher.iterate_metrics(annotation, prediction):
            assert evaluation_result.name == 'accuracy_per_class'
            assert len(evaluation_result.evaluated_value) == 4
            assert evaluation_result.evaluated_value[0] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[1] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[2] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[3] == pytest.approx(0.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None

    def test_classification_per_class_accuracy_prediction_top3(self):
        annotation = [ClassificationAnnotation('identifier_1', 1), ClassificationAnnotation('identifier_2', 1)]
        prediction = [
            ClassificationPrediction('identifier_1', [1.0, 2.0, 3.0, 4.0]),
            ClassificationPrediction('identifier_2', [2.0, 1.0, 3.0, 4.0])
        ]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy_per_class', 'top_k': 3}]}
        dataset = DummyDataset(label_map={0: '0', 1: '1', 2: '2', 3: '3'})
        dispatcher = MetricsExecutor(config, dataset)

        dispatcher.update_metrics_on_batch(annotation, prediction)

        for _, evaluation_result in dispatcher.iterate_metrics(annotation, prediction):
            assert evaluation_result.name == 'accuracy_per_class'
            assert len(evaluation_result.evaluated_value) == 4
            assert evaluation_result.evaluated_value[0] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[1] == pytest.approx(0.5)
            assert evaluation_result.evaluated_value[2] == pytest.approx(0.0)
            assert evaluation_result.evaluated_value[3] == pytest.approx(0.0)
            assert evaluation_result.reference_value is None
            assert evaluation_result.threshold is None


class TestMetricExtraArgs:
    def test_all_metrics_raise_config_error_on_extra_args(self):
        for provider in Metric.providers:
            adapter_config = {'type': provider, 'something_extra': 'extra'}
            with pytest.raises(ConfigError):
                Metric.provide(provider, adapter_config, None)

    def test_detection_recall_raise_config_error_on_extra_args(self):
        adapter_config = {'type': 'recall', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('recall', adapter_config, None)

    def test_detection_miss_rate_raise_config_error_on_extra_args(self):
        adapter_config = {'type': 'miss_rate', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('miss_rate', adapter_config, None)

    def test_accuracy_raise_config_error_on_extra_args(self):
        adapter_config = {'type': 'accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('accuracy', adapter_config, None)

    def test_per_class_accuracy_raise_config_error_on_extra_args(self):
        adapter_config = {'type': 'accuracy_per_class', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('accuracy_per_class', adapter_config, None)

    def test_character_recognition_accuracy_raise_config_error_on_extra_args(self):
        adapter_config = {'type': 'character_recognition_accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('character_recognition_accuracy', adapter_config, None)

    def test_multi_accuracy_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'multi_accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('multi_accuracy', metric_config, None)

    def test_multi_precision_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'multi_precision', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('multi_precision', metric_config, None)

    def test_f1_score_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'f1-score', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('f1-score', metric_config, None)

    def test_mae_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'mae', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('mae', metric_config, None)

    def test_mse_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'mse', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('mse', metric_config, None)

    def test_rmse_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'rmse', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('rmse', metric_config, None)

    def test_mae_on_interval_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'mae_on_interval', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('mae_on_interval', metric_config, None)

    def test_mse_on_interval_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'mse_on_interval', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('mse_on_interval', metric_config, None)

    def test_rmse_on_interval_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'rmse_on_interval', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('rmse_on_interval', metric_config, None)

    def test_per_point_normed_error_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'per_point_normed_error', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('per_point_normed_error', metric_config, None)

    def test_average_point_error_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'normed_error', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('normed_error', metric_config, None)

    def test_reid_cmc_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'cmc', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('cmc', metric_config, None)

    def test_reid_map_raise_config_error_on_extra_args(self):
        adapter_config = {'type': 'reid_map', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('reid_map', adapter_config, None)

    def test_pairwise_accuracy_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'pairwise_accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('pairwise_accuracy', metric_config, None)

    def test_segmentation_accuracy_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'segmentation_accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('segmentation_accuracy', metric_config, None)

    def test_mean_iou_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'mean_iou', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('mean_iou', metric_config, None)

    def test_mean_accuracy_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'mean_accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('mean_accuracy', metric_config, None)

    def test_frequency_weighted_accuracy_raise_config_error_on_extra_args(self):
        metric_config = {'type': 'frequency_weighted_accuracy', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            Metric.provide('frequency_weighted_accuracy', metric_config, None)
