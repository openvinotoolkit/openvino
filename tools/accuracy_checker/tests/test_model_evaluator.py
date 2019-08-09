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

from unittest.mock import Mock, MagicMock

from accuracy_checker.evaluators import ModelEvaluator


class TestModelEvaluator:
    def setup_method(self):
        self.launcher = Mock()
        self.launcher.predict.return_value = []
        data = MagicMock(data=MagicMock(), metadata=MagicMock(), identifier=0)
        self.preprocessor = Mock()
        self.preprocessor.process = Mock(return_value=data)
        self.postprocessor = Mock()
        self.adapter = MagicMock(return_value=[])
        self.input_feeder = Mock()
        self.data_reader = Mock(return_value=data)
        self.data_reader.data_source = 'source'

        annotation_0 = MagicMock()
        annotation_0.identifier = 0
        annotation_0.metadata = {'data_source': MagicMock()}
        annotation_1 = MagicMock()
        annotation_1.identifier = 1
        annotation_1.metadata = {'data_source': MagicMock()}
        annotation_container_0 = MagicMock()
        annotation_container_0.values = MagicMock(return_value=[annotation_0])
        annotation_container_1 = MagicMock()
        annotation_container_1.values = MagicMock(return_value=([annotation_1]))
        self.annotations = [[annotation_container_0], [annotation_container_1]]

        self.dataset = MagicMock()
        self.dataset.__iter__.return_value = self.annotations

        self.postprocessor.process_batch = Mock(side_effect=[
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ])
        self.postprocessor.process_dataset = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))
        self.postprocessor.full_process = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))

        self.metric = Mock()
        self.metric.update_metrics_on_batch = Mock()

        self.evaluator = ModelEvaluator(self.launcher, self.input_feeder, self.adapter, self.data_reader, self.preprocessor, self.postprocessor, self.dataset, self.metric)
        self.evaluator.store_predictions = Mock()
        self.evaluator.load = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))

    def test_process_dataset_without_storing_predictions_and_dataset_processors(self):
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)
        assert self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_without_storing_predictions_and_with_dataset_processors(self):
        self.postprocessor.has_dataset_processors = True

        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_with_storing_predictions_and_without_dataset_processors(self):
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset('path', None)

        assert self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)
        assert self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_with_storing_predictions_and_with_dataset_processors(self):
        self.postprocessor.has_dataset_processors = True

        self.evaluator.process_dataset('path', None)

        assert self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_with_loading_predictions_and_without_dataset_processors(self, mocker):
        mocker.patch('accuracy_checker.evaluators.model_evaluator.get_path')
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset('path', None)

        assert not self.evaluator.store_predictions.called
        assert self.evaluator.load.called
        assert not self.launcher.predict.called
        assert not self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert not self.postprocessor.process_dataset.called
        assert self.postprocessor.full_process.called

    def test_process_dataset_with_loading_predictions_and_with_dataset_processors(self, mocker):
        mocker.patch('accuracy_checker.evaluators.model_evaluator.get_path')
        self.postprocessor.has_dataset_processors = True

        self.evaluator.process_dataset('path', None)

        assert not self.evaluator.store_predictions.called
        assert self.evaluator.load.called
        assert not self.launcher.predict.called
        assert not self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert not self.postprocessor.process_dataset.called
        assert self.postprocessor.full_process.called
