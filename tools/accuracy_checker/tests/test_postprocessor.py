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
import pytest

from accuracy_checker.config import ConfigError
from accuracy_checker.postprocessor import PostprocessingExecutor

from accuracy_checker.representation import (
    DetectionAnnotation,
    DetectionPrediction,
    ContainerAnnotation,
    ContainerPrediction,
    ClassificationAnnotation
)

from .common import make_representation, make_segmentation_representation


def postprocess_data(executor, annotations, predictions):
    return executor.full_process(annotations, predictions)


class TestPostprocessor:
    def test_without_apply_to_and_sources_filter_raise_config_error_exception(self):
        config = [{'type': 'filter', 'labels': [1]}]

        with pytest.raises(ConfigError):
            PostprocessingExecutor(config)

    def test_both_provided_apply_to_and_sources_filter_raise_config_error_exception(self):
        config = [{
            'type': 'filter',
            'apply_to': 'prediction',
            'annotation_source': 'annotation',
            'labels': [1]
        }]

        with pytest.raises(ConfigError):
            PostprocessingExecutor(config)

    def test_filter_annotations_unsupported_source_type_in_container_raise_type_error_exception(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation', 'labels': [1]}]
        annotation = ContainerAnnotation({'annotation': ClassificationAnnotation()})
        executor = PostprocessingExecutor(config)

        with pytest.raises(TypeError):
            postprocess_data(executor, [annotation], [None])

    def test_filter_annotations_source_not_found_raise_config_error_exception(self):
        config = [{'type': 'filter', 'annotation_source': 'ann', 'labels': [1]}]
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        executor = PostprocessingExecutor(config)

        with pytest.raises(ConfigError):
            postprocess_data(executor, [annotation], [None])

    def test_filter_predictions_unsupported_source_type_raise_type_error_exception(self):
        config = [{
            'type': 'filter',
            'prediction_source': 'detection_out',
            'labels': [1],
            'remove_filtered': False
        }]
        prediction = ContainerPrediction({'detection_out': ClassificationAnnotation()})
        executor = PostprocessingExecutor(config)

        with pytest.raises(TypeError):
            postprocess_data(executor, [None], [prediction])

    def test_filter_predictions_source_not_found_raise_config_error_exception(self):
        config = [{
            'type': 'filter', 'prediction_source': 'undefined', 'labels': [1]
        }]
        prediction = ContainerPrediction({'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]})
        executor = PostprocessingExecutor(config)

        with pytest.raises(ConfigError):
            postprocess_data(executor, [None], [prediction])

    def test_filter_container_annotations_by_labels_with_ignore_using_source(self):
        config = [{
            'type': 'filter', 'annotation_source': 'annotation', 'labels': [1], 'remove_filtered': False
        }]
        annotation = ContainerAnnotation({
            'annotation':  make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
            )[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_container_annotations_by_labels_with_ignore_using_apply_to(self):
        config = [{
            'type': 'filter',
            'apply_to': 'annotation',
            'labels': [1],
            'remove_filtered': False
        }]
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
            )[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_regular_annotations_by_labels_with_ignore(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': [1], 'remove_filtered': False}]
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected = make_representation(
                '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
            )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_annotations_by_labels_with_ignore(self):
        config = [{
            'type': 'filter',
            'annotation_source': ['annotation1', 'annotation2'],
            'labels': [1],
            'remove_filtered': False
        }]
        annotation = ContainerAnnotation({
            'annotation1': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0],
            'annotation2': make_representation('1 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation1': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
            )[0],
            'annotation2': make_representation(
                '1 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [0, 1]}]
            )[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_annotations_by_labels_with_ignore_using_apply_to(self):
        config = [{
            'type': 'filter',
            'apply_to': 'annotation',
            'labels': [1],
            'remove_filtered': False
        }]
        annotation = ContainerAnnotation({
            'annotation1': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0],
            'annotation2': make_representation('1 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation1': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
            )[0],
            'annotation2': make_representation(
                '1 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [0, 1]}]
            )[0]
        })
        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_regular_annotations_by_labels_with_remove_using_annotation_source_warm_user_warning(self):
        config = [{
            'type': 'filter',
            'annotation_source': 'annotation',
            'labels': [1],
            'remove_filtered': True
        }]
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected = make_representation('0 0 0 10 10', is_ground_truth=True)[0]

        with pytest.warns(UserWarning):
            postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_regular_annotations_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': [1], 'remove_filtered': True}]
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected = make_representation('0 0 0 10 10', is_ground_truth=True)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_annotations_by_labels_with_remove_on_container(self):
        config = [{
            'type': 'filter',
            'annotation_source': 'annotation',
            'labels': [1],
            'remove_filtered': True
        }]
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_annotations_by_labels_with_remove_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': [1], 'remove_filtered': True}]
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_annotations_by_labels_with_remove(self):
        config = [{
            'type': 'filter',
            'annotation_source': ['annotation1', 'annotation2'],
            'labels': [1], 'remove_filtered': True
        }]
        annotation = ContainerAnnotation({
            'annotation1': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0],
            'annotation2': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation1': make_representation('0 0 0 10 10', is_ground_truth=True)[0],
            'annotation2': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_by_labels_with_remove_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': [1], 'remove_filtered': True}]
        annotation = ContainerAnnotation({
            'annotation1': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0],
            'annotation2': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })
        expected = ContainerAnnotation({
            'annotation1': make_representation('0 0 0 10 10', is_ground_truth=True)[0],
            'annotation2': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_predictions_by_labels_with_ignore(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected = DetectionPrediction(labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_ignore_on_container(self):
        config = [{
            'type': 'filter',
            'prediction_source': 'detection_out',
            'labels': [1],
            'remove_filtered': False
        }]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected = ContainerPrediction({'detection_out': make_representation(
            '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
        )[0]})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_ignore_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': [1], 'remove_filtered': False}]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected = ContainerPrediction({'detection_out': make_representation(
            '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
        )[0]})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_ignore(self):
        config = [{
            'type': 'filter', 'prediction_source': ['detection_out1', 'detection_out2'], 'labels': [1],
            'remove_filtered': False
        }]
        prediction = ContainerPrediction({
            'detection_out1': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0],
            'detection_out2': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected = ContainerPrediction({
            'detection_out1': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
            )[0],
            'detection_out2': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
            )[0]
        })

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{
            'type': 'filter', 'apply_to': 'prediction', 'labels': [1], 'remove_filtered': False
        }]
        prediction = ContainerPrediction({
            'detection_out1': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0],
            'detection_out2': make_representation('1 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected = ContainerPrediction({
            'detection_out1': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
            )[0],
            'detection_out2': make_representation(
                '1 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [0, 1]}]
            )[0]
        })

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': [1], 'remove_filtered': True}]
        prediction = make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)
        expected = make_representation('0 0 0 10 10', score=1)

        postprocess_data(PostprocessingExecutor(config), [None], prediction)

        assert prediction == expected

    def test_filter_predictions_by_labels_with_remove_on_container(self):
        config = [{
            'type': 'filter', 'prediction_source': 'detection_out', 'labels': [0], 'remove_filtered': True
        }]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected = ContainerPrediction({'detection_out':  make_representation('1 0 0 11 11', score=1)[0]})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_remove_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': [0], 'remove_filtered': True}]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected = ContainerPrediction({'detection_out': make_representation('1 0 0 11 11', score=1)[0]})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_remove(self):
        config = [{
            'type': 'filter',
            'prediction_source': ['detection_out1', 'detection_out2'],
            'labels': [1],
            'remove_filtered': True
        }]
        prediction = ContainerPrediction({
            'detection_out1': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0],
            'detection_out2': make_representation('0 0 0 10 10', score=1)[0]
        })
        expected = ContainerPrediction({
            'detection_out1': make_representation('0 0 0 10 10', score=1)[0],
            'detection_out2': make_representation('0 0 0 10 10', score=1)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': [1], 'remove_filtered': True}]
        prediction = ContainerPrediction({
            'detection_out1': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0],
            'detection_out2': make_representation('0 0 0 10 10', score=1)[0]
        })
        expected = ContainerPrediction({
            'detection_out1': make_representation('0 0 0 10 10', score=1)[0],
            'detection_out2': make_representation('0 0 0 10 10', score=1)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_regular_annotations_and_regular_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': False}]
        prediction = make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        expected_prediction = make_representation(
            '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
        )[0]
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected_annotation = make_representation(
            '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_regular_annotations_and_regular_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': True}]
        prediction = make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)
        expected_prediction = make_representation('0 0 0 10 10', score=1)
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)
        expected_annotation = make_representation('0 0 0 10 10', is_ground_truth=True)

        postprocess_data(PostprocessingExecutor(config), annotation, prediction)

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_regular_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': False}]
        prediction = make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        expected_prediction = make_representation(
            '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
        )[0]
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected_annotation = make_representation(
            '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_regular_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': True}]
        prediction = make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        expected_prediction = make_representation('0 0 0 10 10', score=1)[0]
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected_annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_regular_annotations_and_container_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': False}]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected_prediction = ContainerPrediction({
            'detection_out': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
            )[0]
        })
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected_annotation = make_representation(
            '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_regular_annotations_and_container_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': True}]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected_prediction = ContainerPrediction({'detection_out': make_representation('0 0 0 10 10', score=1)[0]})
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected_annotation = make_representation('0 0 0 10 10', is_ground_truth=True)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': False}]
        prediction = ContainerPrediction({
            'detection_out': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected_prediction = ContainerPrediction({
            'detection_out': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}]
            )[0]
        })
        annotation = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        expected_annotation = make_representation(
            '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': True}]
        prediction = ContainerPrediction({
            'prediction': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]
        })
        expected_prediction = ContainerPrediction({'prediction': make_representation('0 0 0 10 10', score=1)[0]})
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected_annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10', is_ground_truth=True)[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_ignore_using_sources(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': [1], 'remove_filtered': False}]
        prediction = ContainerPrediction({'prediction': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]})
        expected_prediction = ContainerPrediction({
            'prediction': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1, meta=[{'difficult_boxes': [1]}])[0]
        })
        annotation = ContainerAnnotation({
            'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]
        })
        expected_annotation = ContainerAnnotation({
            'annotation': make_representation(
                '0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True, meta=[{'difficult_boxes': [1]}]
            )[0]
        })

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_remove_using_sources(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation', 'prediction_source': 'prediction',
                   'labels': [1], 'remove_filtered': True}]
        prediction = ContainerPrediction({'prediction': make_representation('0 0 0 10 10; 1 0 0 11 11', score=1)[0]})
        expected_prediction = ContainerPrediction({'prediction': make_representation('0 0 0 10 10', score=1)[0]})
        annotation = ContainerAnnotation(
            {'annotation': make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)[0]})
        expected_annotation = ContainerAnnotation(
            {'annotation': make_representation('0 0 0 10 10', is_ground_truth=True)[0]})

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_annotations_by_min_confidence_do_nothing(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_confidence': 0.5, 'remove_filtered': True}]
        annotations = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)
        expected_annotations = make_representation('0 0 0 10 10; 1 0 0 11 11', is_ground_truth=True)

        postprocess_data(PostprocessingExecutor(config), annotations, [None])

        assert np.array_equal(annotations, expected_annotations)

    def test_filter_predictions_by_min_confidence_with_ignore(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_confidence': 0.5, 'remove_filtered': False}]
        predictions = [
            make_representation('0 0 0 10 10; 1 0 0 11 11', score=[0.3, 0.8])[0],
            make_representation('0 0 0 10 10; 1 0 0 11 11', score=[0.5, 0.4])[0]
        ]
        expected_predictions = [
            make_representation('0 0 0 10 10; 1 0 0 11 11', score=[0.3, 0.8], meta=[{'difficult_boxes': [0]}])[0],
            make_representation('0 0 0 10 10; 1 0 0 11 11', score=[0.5, 0.4], meta=[{'difficult_boxes': [1]}])[0]
        ]

        executor = PostprocessingExecutor(config)
        postprocess_data(executor, [None, None], predictions)

        assert np.array_equal(predictions, expected_predictions)

    def test_filter_predictions_by_min_confidence_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_confidence': 0.5, 'remove_filtered': True}]
        predictions = [
            make_representation('0 0 0 10 10; 1 0 0 11 11', score=[0.3, 0.8])[0],
            make_representation('0 0 0 10 10; 1 0 0 11 11', score=[0.5, 0.4])[0]
        ]
        expected_predictions = [
            make_representation('1 0 0 11 11', score=0.8)[0],
            make_representation('0 0 0 10 10', score=0.5)[0]
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected_predictions)

    def test_filter_annotations_by_height_range_with_ignored(self):
        config = [{
            'type': 'filter',
            'apply_to': 'annotation',
            'height_range': '(10.0, 20.0)',
            'remove_filtered': False
        }]
        annotations = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True)[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', is_ground_truth=True)[0]
        ]
        expected = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True, meta=[{'difficult_boxes': [1]}])[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', is_ground_truth=True, meta=[{'difficult_boxes': [0, 1]}])[0]
        ]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected)

    def test_filter_annotations_by_height_range_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'height_range': '(10.0, 20.0)', 'remove_filtered': True}]
        annotations = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True)[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', is_ground_truth=True)[0]
        ]
        expected = [
            make_representation('0 0 5 0 15', is_ground_truth=True)[0],
            make_representation('', is_ground_truth=True)[0]
        ]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected)

    def test_filter_predictions_by_height_range_with_ignored(self):
        config = [{
            'type': 'filter',
            'apply_to': 'prediction',
            'height_range': '(10.0, 20.0)',
            'remove_filtered': False
        }]
        predictions = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', score=1)[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', score=1)[0]
        ]
        expected = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', score=1, meta=[{'difficult_boxes': [1]}])[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', score=1, meta=[{'difficult_boxes': [0, 1]}])[0]
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected)

    def test_filter_predictions_by_height_range_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'height_range': '(10.0, 20.0)', 'remove_filtered': True}]
        predictions = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', score=1)[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', score=1)[0]
        ]
        expected = [
            make_representation('0 0 5 0 15', score=1)[0],
            make_representation('', score=1)[0]
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected)

    def test_filter_predictions_by_unknown_min_visibility_raises_value_error_exception(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_visibility': 'unknown'}]
        predictions = [
           make_representation('0 0 5 0 15; 1 0 10 0 15', score=1)[0],
           make_representation('0 0 5 0 35; 1 0 10 0 40', score=1)[0]
        ]

        with pytest.raises(ValueError):
            postprocess_data(PostprocessingExecutor(config), [None], predictions)

    def test_filter_annotations_by_unknown_min_visibility_raises_value_error_exception(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'unknown'}]
        annotations = [DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0])]

        with pytest.raises(ValueError):
            postprocess_data(PostprocessingExecutor(config), annotations, [None])

    def test_filter_predictions_by_visibility_raises_value_error_with_unknown_visibility(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_visibility': 'heavy occluded'}]
        predictions = [DetectionPrediction(
            y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'visibilities': ['unknown']}
        )]

        with pytest.raises(ValueError):
            postprocess_data(PostprocessingExecutor(config), [None], predictions)

    def test_filter_annotations_by_visibility_raises_value_error_with_unknown_visibility(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'heavy occluded'}]
        annotations = [DetectionAnnotation(
            y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'visibilities': ['unknown']}
        )]

        with pytest.raises(ValueError):
            postprocess_data(PostprocessingExecutor(config), annotations, [None])

    def test_filter_by_visibility_does_nothing_with_annotations_without_visibility(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'heavy occluded'}]
        annotations = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True)[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', is_ground_truth=True)[0]
        ]
        expected = [
            make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True, meta=[{'difficult_boxes': []}])[0],
            make_representation('0 0 5 0 35; 1 0 10 0 40', is_ground_truth=True, meta=[{'difficult_boxes': []}])[0]
        ]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected)

    def test_filter_by_visibility_does_nothing_with_predictions_without_visibility(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_visibility': 'heavy occluded'}]
        predictions = [
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0]),
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0])
        ]
        expected = [
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'difficult_boxes': []}),
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0], metadata={'difficult_boxes': []})
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected)

    def test_filter_by_visibility_does_nothing_with_default_visibility_level_and_heavy_occluded(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'heavy occluded'}]
        annotation = make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True)[0]
        expected = make_representation(
            '0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True, meta=[{'difficult_boxes': []}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_by_visibility_does_nothing_with_default_visibility_level_and_partially_occluded(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'partially occluded'}]
        annotation = make_representation('0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True)[0]
        expected = make_representation(
            '0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True, meta=[{'difficult_boxes': []}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_by_visibility_filters_partially_occluded_remove_filtered(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'partially occluded',
                   'remove_filtered': True}]
        annotation = make_representation(
            '0 0 5 0 15; 1 0 10 0 15', is_ground_truth=True,
            meta=[{'visibilities': ['heavy occluded', 'partially occluded']}]
        )[0]
        expected = make_representation(
            '1 0 10 0 15', is_ground_truth=True, meta=[{'visibilities': ['heavy occluded', 'partially occluded']}]
        )[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_nms(self, mocker):
        mock = mocker.patch('accuracy_checker.postprocessor.nms.NMS.process_all', return_value=([], []))
        config = [{'type': 'nms', 'overlap': 0.4}]
        postprocess_data(PostprocessingExecutor(config), [], [])
        mock.assert_called_once_with([], [])

    def test_resize_prediction_boxes(self):
        config = [{'type': 'resize_prediction_boxes'}]
        annotation = DetectionAnnotation(metadata={'image_size': [(100, 100, 3)]})
        prediction = make_representation('0 0 0 5 5; 1 7 7 8 8', score=1)[0]
        expected = make_representation('0 0 0 500 500; 1 700 700 800 800', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected

    def test_clip_annotation_denormalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': False}]
        meta = {'image_size': [(10, 10, 3)]}
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True, meta=[meta])[0]
        expected = make_representation('0 0 0 5 5; 1 9 10 10 10', is_ground_truth=True, meta=[meta])[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_annotation_normalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': True}]
        meta = {'image_size': [(10, 10, 3)]}
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True, meta=[meta])[0]
        expected = make_representation('0 0 0 1 1; 1 1 1 1 1', is_ground_truth=True, meta=[meta])[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_annotation_denormalized_boxes_with_size(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': False, 'size': 10}]
        meta = {'image_size': [(10, 10, 3)]}
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True, meta=[meta])[0]
        expected = make_representation('0 0 0 5 5; 1 9 10 10 10', is_ground_truth=True, meta=[meta])[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_annotation_normalized_boxes_with_size_as_normalized(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': True, 'size': 10}]
        meta = {'image_size': [(10, 10, 3)]}
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True, meta=[meta])[0]
        expected = make_representation('0 0 0 1 1; 1 1 1 1 1', is_ground_truth=True, meta=[meta])[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_prediction_denormalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': False}]
        annotation = DetectionAnnotation(metadata={'image_size': [(10, 10, 3)]})
        prediction = make_representation('0 -1 0 5 5; 1 9 11 10 10', score=1)[0]
        expected = make_representation('0 0 0 5 5; 1 9 10 10 10', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected

    def test_clip_prediction_normalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': True}]
        annotation = DetectionAnnotation(metadata={'image_size': [(10, 10, 3)]})
        prediction = make_representation('0 -1 0 5 5; 1 9 11 10 10', score=1)[0]
        expected = make_representation('0 0 0 1 1; 1 1 1 1 1', score=1)[0]
        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected

    def test_clip_predictions_denormalized_boxes_with_size(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': False, 'size': 10}]
        annotation = DetectionAnnotation(metadata={'image_size': [(10, 10, 3)]})
        prediction = make_representation('0 -1 0 5 5; 1 9 11 10 10', score=1)[0]
        expected = make_representation('0 0 0 5 5; 1 9 10 10 10', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected

    def test_clip_predictions_normalized_boxes_with_size_as_normalized(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': True, 'size': 10}]
        annotation = DetectionAnnotation(metadata={'image_size': [(10, 10, 3)]})
        prediction = make_representation('0 -1 0 5 5; 1 9 11 10 10', score=1)[0]
        expected = make_representation('0 0 0 1 1; 1 1 1 1 1', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected

    def test_cast_to_int_default(self):
        config = [{'type': 'cast_to_int'}]
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        prediction = make_representation('0 -1.1 0.5 5.9 5.1; 1 -9.9 11.5 10.9 10.1', score=1)[0]
        expected_annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        expected_prediction = make_representation('0 -1 0 6 5; 1 -10 12 11 10', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_nearest(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'nearest'}]
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        prediction = make_representation('0 -1.1 0.5 5.9 5.1; 1 -9.9 11.5 10.9 10.1', score=1)[0]
        expected_annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        expected_prediction = make_representation('0 -1 0 6 5; 1 -10 12 11 10', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_nearest_to_zero(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'nearest_to_zero'}]
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        prediction = make_representation('0 -1.1 0.5 5.9 5.1; 1 -9.9 11.5 10.9 10.1', score=1)[0]
        expected_annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        expected_prediction = make_representation('0 -1 0 5 5; 1 -9 11 10 10', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_lower(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'lower'}]
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        prediction = make_representation('0 -1.1 0.5 5.9 5.1; 1 -9.9 11.5 10.9 10.1', score=1)[0]
        expected_annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        expected_prediction = make_representation('0 -2 0 5 5; 1 -10 11 10 10', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_greater(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'greater'}]
        annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        prediction = make_representation('0 -1.1 0.5 5.9 5.1; 1 -9.9 11.5 10.9 10.1', score=1)[0]
        expected_annotation = make_representation('0 -1 0 5 5; 1 9 11 10 10', is_ground_truth=True)[0]
        expected_prediction = make_representation('0 -1 1 6 6; 1 -9 12 11 11', score=1)[0]

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_unknown_raise_config_error(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'unknown'}]

        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_extend_segmentation_mask_with_float_filling_raise_config_error(self):
        config = [{'type': 'extend_segmentation_mask', 'filling_label':  0.5}]

        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_extend_segmentation_mask_default(self):
        config = [{'type': 'extend_segmentation_mask'}]
        annotation = make_segmentation_representation(np.zeros((5, 5)), ground_truth=True)
        prediction = make_segmentation_representation(np.zeros((7, 7)), ground_truth=False)
        expected_annotation_mask = np.zeros((7, 7))
        expected_annotation_mask[0, :] = 255
        expected_annotation_mask[:, 0] = 255
        expected_annotation_mask[-1, :] = 255
        expected_annotation_mask[:, -1] = 255
        expected_prediction_mask = np.zeros((7, 7))
        postprocess_data(PostprocessingExecutor(config), annotation, prediction)
        assert np.array_equal(prediction[0].mask, expected_prediction_mask)
        assert np.array_equal(annotation[0].mask, expected_annotation_mask)

    def test_extend_segmentation_mask_do_nothing(self):
        config = [{'type': 'extend_segmentation_mask'}]
        annotation = make_segmentation_representation(np.zeros((5, 5)), ground_truth=True)
        prediction = make_segmentation_representation(np.zeros((5, 5)), ground_truth=False)
        expected_mask = np.zeros((5, 5))
        postprocess_data(PostprocessingExecutor(config), annotation, prediction)
        assert np.array_equal(prediction[0].mask, expected_mask)
        assert np.array_equal(annotation[0].mask, expected_mask)

    def test_extend_segmentation_mask_asymmetrical(self):
        config = [{'type': 'extend_segmentation_mask'}]
        annotation = make_segmentation_representation(np.zeros((5, 5)), ground_truth=True)
        prediction = make_segmentation_representation(np.zeros((6, 7)), ground_truth=False)
        expected_annotation_mask = np.zeros((6, 7))
        expected_annotation_mask[:, 0] = 255
        expected_annotation_mask[-1, :] = 255
        expected_annotation_mask[:, -1] = 255
        expected_prediction_mask = np.zeros((6, 7))
        postprocess_data(PostprocessingExecutor(config), annotation, prediction)
        assert np.array_equal(prediction[0].mask, expected_prediction_mask)
        assert np.array_equal(annotation[0].mask, expected_annotation_mask)

    def test_extend_segmentation_mask_raise_config_error_if_prediction_less_annotation(self):
        config = [{'type': 'extend_segmentation_mask'}]
        annotation = make_segmentation_representation(np.zeros((5, 5)), ground_truth=True)
        prediction = make_segmentation_representation(np.zeros((4, 4)), ground_truth=False)
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), annotation, prediction)

    def test_extend_segmentation_mask_with_filling_label(self):
        config = [{'type': 'extend_segmentation_mask', 'filling_label': 1}]
        annotation = make_segmentation_representation(np.zeros((5, 5)), ground_truth=True)
        prediction = make_segmentation_representation(np.zeros((7, 7)), ground_truth=False)
        expected_annotation_mask = np.zeros((7, 7))
        expected_annotation_mask[0, :] = 1
        expected_annotation_mask[:, 0] = 1
        expected_annotation_mask[-1, :] = 1
        expected_annotation_mask[:, -1] = 1
        expected_prediction_mask = np.zeros((7, 7))
        postprocess_data(PostprocessingExecutor(config), annotation, prediction)
        assert np.array_equal(prediction[0].mask, expected_prediction_mask)
        assert np.array_equal(annotation[0].mask, expected_annotation_mask)


class TestPostprocessorExtraArgs:
    def test_cast_to_int_raise_config_error_on_extra_args(self):
        config = {'type': 'cast_to_int', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_clip_boxes_raise_config_error_on_extra_args(self):
        config = {'type': 'clip_boxes', 'size': 1, 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_correct_yolo_v2_boxes_raise_config_error_on_extra_args(self):
        config = {'type': 'correct_yolo_v2_boxes', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_encode_segmentation_mask_raise_config_error_on_extra_args(self):
        config = {'type': 'encode_segmentation_mask', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_filter_raise_config_error_on_extra_args(self):
        config = {'type': 'filter', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_nms_raise_config_error_on_extra_args(self):
        config = {'type': 'nms', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_normalize_landmarks_points_raise_config_error_on_extra_args(self):
        config = {'type': 'normalize_landmarks_points', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_resize_prediction_boxes_raise_config_error_on_extra_args(self):
        config = {'type': 'resize_prediction_boxes', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_resize_segmentation_mask_raise_config_error_on_extra_args(self):
        config = {'type': 'resize_segmentation_mask', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])

    def test_extend_segmentation_mask_raise_config_error_on_extra_args(self):
        config = {'type': 'resize_segmentation_mask', 'something_extra': 'extra'}
        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])
