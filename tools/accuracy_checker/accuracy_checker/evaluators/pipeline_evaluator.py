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

from collections import OrderedDict
import numpy as np

from ..pipeline_connectors import create_connection_description
from ..utils import get_indexs, find_nearest
from ..adapters import create_adapter
from ..data_readers import create_reader
from ..dataset import Dataset
from ..launcher import create_launcher, InputFeeder
from ..metrics import MetricsExecutor
from ..pipeline_connectors import StageConnectionDescription, Connection
from ..postprocessor import PostprocessingExecutor
from..preprocessor import PreprocessingExecutor


def get_processing_info(pipeline_config):
    name = pipeline_config['name']
    stages = pipeline_config['stages']
    dataset_name = stages[0]['dataset']['name']
    launcher = {}
    for stage in stages:
        if 'launcher' in stage:
            launcher = stage['launcher']
            break
    framework = launcher.get('framework')
    device = launcher.get('device')
    tags = launcher.get('tags')

    return name, framework, device, tags, dataset_name


def create_launcher_attribution(launchers_ids, launchers, datasets_ids, datasets, executors, executor_types):
    launchers_ids = np.array(launchers_ids)
    datasets_ids = np.array(datasets_ids)
    for launcher_id_info, launcher in zip(enumerate(launchers_ids), launchers):
        iteration, launcher_id = launcher_id_info
        input_feeder = InputFeeder(
            launcher.config.get('inputs', []), launcher.get_all_inputs(), launcher.fit_to_input
        )
        launchers_ids[iteration:] += 1
        executors.insert(launcher_id, input_feeder)
        executor_types.insert(launcher_id, 'input_feeder')
        adapter_config = launcher.config.get('adapter')
        dataset_id = find_nearest(datasets_ids, launcher_id, 'less')
        datasets_ids[dataset_id + 1:] += 1
        dataset = datasets[dataset_id] if dataset_id != -1 else None
        launcher_id += 1
        if adapter_config:
            adapter = create_adapter(adapter_config, launcher, dataset)
            executors.insert(launcher_id + 1, adapter)
            executor_types.insert(launcher_id + 1, 'adapter')
            if dataset_id != datasets_ids.size - 1:
                datasets_ids[dataset_id + 1:] += 1
            if iteration != launchers_ids.size - 1:
                launchers_ids[iteration + 1:] += 1


def set_metrics_dataset(metrics_ids, metrics_executors, datasets_ids, datasets):
    for metrics_id, metric_executor in zip(metrics_ids, metrics_executors):
        dataset_id = find_nearest(datasets_ids, metrics_id, 'less')
        if dataset_id != -1:
            metric_executor.dataset = datasets[dataset_id].metadata


class PipeLineStage:
    def __init__(self, evaluation_context, executors):
        self._evaluation_context = evaluation_context
        self.executors = executors

    def run(self):
        for executor in self.executors:
            executor(self.evaluation_context)

    @classmethod
    def from_configs(cls, stage_name, stage_config):
        config_mapping = {
            'dataset': Dataset,
            'preprocessing': PreprocessingExecutor,
            'launcher': create_launcher,
            'postprocessing': PostprocessingExecutor,
            'metrics': MetricsExecutor,
            'reader': create_reader,
        }

        executor_types = []
        executors = []
        for key, config in stage_config.items():
            if key in config_mapping:
                connection = create_connection_description(config, stage_name)
                if connection:
                    executors.append(connection)
                    executor_types.append('connection')
                executor_creator = config_mapping[key]
                executor = executor_creator(config)
                executor_types.append(key)
                executors.append(executor)

        dataset_ids = get_indexs(executor_types, 'dataset')
        datasets = [executors[idx] for idx in dataset_ids]
        launcher_ids = get_indexs(executor_types, 'launcher')
        launchers = [executors[idx] for idx in launcher_ids]
        create_launcher_attribution(launcher_ids, launchers, dataset_ids, datasets, executors, executor_types)

        metrics_executors_id = get_indexs(executor_types, 'metrics')
        dataset_ids = get_indexs(executor_types, 'dataset')
        metrics_executors = [executors[idx] for idx in metrics_executors_id]
        set_metrics_dataset(metrics_executors_id, metrics_executors, dataset_ids, datasets)
        dataset = datasets[0] if datasets else None
        eval_context = EvaluationContext(dataset, metrics_executors, launchers)

        return cls(eval_context, executors)

    @property
    def evaluation_context(self):
        return self._evaluation_context

    @evaluation_context.setter
    def evaluation_context(self, new_context):
        _shared_context = new_context.shared_context
        for field, value in _shared_context.items():
            if value:
                setattr(self._evaluation_context, field, value)


class EvaluationContext:
    def __init__(self, dataset, metric_executor=None, launcher=None):
        self.annotations = []
        self.predictions = []
        self.annotation_batch = []
        self.prediction_batch = []
        self.data_batch = []
        self.metrics_results = []
        self.identifiers_batch = []
        self.metrics_executor = metric_executor
        self.dataset_size = dataset.size if dataset else 0
        self.launcher = launcher
        self.dataset = dataset

    @property
    def shared_context(self):
        _shared_context = {
            'annotations': self.annotations,
            'predictions': self.predictions,
            'annotation_batch': self.annotation_batch,
            'prediction_batch': self.prediction_batch,
            'data_batch': self.data_batch,
            'identifiers_batch': self.identifiers_batch
        }
        return _shared_context


class PipeLineEvaluator:
    def __init__(self, stages):
        self.stages = stages
        self.create_connectors()
        self.context = next(iter(stages.values())).evaluation_context

    @classmethod
    def from_configs(cls, pipeline_config):
        stages = OrderedDict()
        for stage_config in pipeline_config:
            stage_name = stage_config['stage']
            evaluation_stage = PipeLineStage.from_configs(stage_name, stage_config)
            stages[stage_name] = evaluation_stage
        return cls(stages)

    def create_connectors(self):
        def make_connection(stages, connection_template):
            return Connection(stages, connection_template)

        def replace_connections(stage, all_stages):
            for executor_id, executor in enumerate(stage.executors):
                if isinstance(executor, StageConnectionDescription):
                    connector = make_connection(all_stages, executor)
                    stage.executors[executor_id] = connector

        for _, stage in self.stages.items():
            replace_connections(stage, self.stages)

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        self.progress_reporter = progress_reporter
        dataset_size = self.context.dataset_size
        dataset_size = dataset_size if dataset_size else 0
        self.progress_reporter.reset(dataset_size)
        iteration = 0
        previous_context = self.context
        while self.progress_reporter.progress != 100:
            for _, stage in self.stages.items():
                stage.evaluation_context = previous_context
                stage.run()
                previous_context = stage.evaluation_context
            iteration += 1
            progress_reporter.update(iteration, len(previous_context.data_batch))
        self.context = previous_context

        if progress_reporter:
            progress_reporter.finish()

    def compute_metrics(self, output_callback=None, ignore_results_formatting=False):
        def eval_metrics(metrics_executor, annotations, predictions):
            for result_presenter, evaluated_metric in metrics_executor.iterate_metrics(annotations, predictions):
                result_presenter.write_result(evaluated_metric, output_callback, ignore_results_formatting)

        for _, stage in self.stages.items():
            metrics_executors = stage.evaluation_context.metrics_executor
            for metrics_executor in metrics_executors:
                eval_context = stage.evaluation_context
                eval_metrics(metrics_executor, eval_context.annotations, eval_context.predictions)

    def release(self):
        for _, stage in self.stages.items():
            for launcher in stage.evaluation_context.launcher:
                launcher.release()
