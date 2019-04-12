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

import copy
import pickle

from .utils import get_path
from .dataset import Dataset
from .launcher import create_launcher, DummyLauncher
from .launcher.loaders import PickleLoader
from .logging import print_info
from .metrics import MetricsExecutor
from .postprocessor import PostprocessingExecutor
from .preprocessor import PreprocessingExecutor


class ModelEvaluator:
    def __init__(self, launcher, preprocessor, postprocessor, dataset, metric):
        self.launcher = launcher
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataset = dataset
        self.metric_executor = metric

        self._annotations = []
        self._predictions = []

    @classmethod
    def from_configs(cls, launcher_config, dataset_config):
        dataset_name = dataset_config['name']
        preprocessor = PreprocessingExecutor(dataset_config.get('preprocessing'), dataset_name)
        dataset = Dataset(dataset_config, preprocessor)

        launcher = create_launcher(launcher_config, dataset.metadata)
        postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset.metadata)
        metric_dispatcher = MetricsExecutor(dataset_config, dataset)

        return cls(launcher, preprocessor, postprocessor, dataset, metric_dispatcher)

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        if self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher):
            self._annotations, self._predictions = self.load(stored_predictions, progress_reporter)
            self._annotations, self._predictions = self.postprocessor.full_process(self._annotations, self._predictions)

            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)
            return self._annotations, self._predictions

        self.dataset.batch = self.launcher.batch
        predictions_to_store = []
        for batch_id, (batch_annotation, batch_input) in enumerate(self.dataset):
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_predictions = self.launcher.predict(batch_identifiers, batch_input, *args, **kwargs)

            if stored_predictions:
                predictions_to_store.extend(copy.deepcopy(batch_predictions))

            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions)
            if not self.postprocessor.has_dataset_processors:
                self.metric_executor.update_metrics_on_batch(annotations, predictions)

            self._annotations.extend(annotations)
            self._predictions.extend(predictions)

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            self.store_predictions(stored_predictions, predictions_to_store)

        if self.postprocessor.has_dataset_processors:
            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)

        return self.postprocessor.process_dataset(self._annotations, self._predictions)

    @staticmethod
    def _is_stored(stored_predictions=None):
        if not stored_predictions:
            return False

        try:
            get_path(stored_predictions)
            return True
        except OSError:
            return False

    def compute_metrics(self, output_callback=None, ignore_results_formatting=False):
        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            result_presenter.write_result(evaluated_metric, output_callback, ignore_results_formatting)

    def load(self, stored_predictions, progress_reporter):
        self._annotations = self.dataset.annotation
        launcher = self.launcher
        if not isinstance(launcher, DummyLauncher):
            launcher = DummyLauncher({
                'framework': 'dummy',
                'loader': PickleLoader.__provider__,
                'data_path': stored_predictions
            }, adapter=None)

        predictions = launcher.predict([annotation.identifier for annotation in self._annotations])

        if progress_reporter:
            progress_reporter.finish(False)

        return self._annotations, predictions

    @staticmethod
    def store_predictions(stored_predictions, predictions):
        # since at the first time file does not exist and then created we can not use it as a pathlib.Path object
        with open(stored_predictions, "wb") as content:
            pickle.dump(predictions, content)
            print_info("prediction objects are save to {}".format(stored_predictions))

    def release(self):
        self.launcher.release()
