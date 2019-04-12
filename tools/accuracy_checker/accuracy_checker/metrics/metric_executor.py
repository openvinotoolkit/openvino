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

from collections import namedtuple

from ..presenters import BasePresenter, EvaluationResult
from ..config import StringField
from ..utils import zipped_transform
from .metric import BaseMetricConfig, Metric
from ..config import ConfigError

MetricInstance = namedtuple('MetricInstance', ['name', 'metric_fn', 'reference', 'threshold', 'presenter'])


class MetricConfig(BaseMetricConfig):
    type = StringField(choices=Metric.providers)


class MetricsExecutor:
    """
    Class for evaluating metrics according to dataset configuration entry.
    """

    def __init__(self, dataset_config, dataset, state=None):
        dataset_name = dataset_config.get('name', '')
        message_prefix = '{}'.format(dataset_name)

        self.state = state or {}
        self._token = 'metrics'

        dataset_metrics = dataset_config.get(self._token)
        if not dataset_metrics:
            raise ConfigError('{} dataset config must specify "{}"'.format(message_prefix, self._token))

        self.dataset = dataset

        self.metrics = []
        type_ = 'type'
        identifier = 'name'
        reference = 'reference'
        threshold = 'threshold'
        presenter = 'presenter'

        for metric_config_entry in dataset_metrics:
            metric_config = MetricConfig(
                "{}.metrics".format(dataset_name), on_extra_argument=MetricConfig.IGNORE_ON_EXTRA_ARGUMENT
            )
            metric_type = metric_config_entry.get(type_)
            metric_config.validate(metric_config_entry, type_)

            metric_identifier = metric_config_entry.get(identifier, metric_type)

            metric_fn = Metric.provide(
                metric_type, metric_config_entry, self.dataset, metric_identifier, state=self.state
            )
            metric_presenter = BasePresenter.provide(metric_config_entry.get(presenter, 'print_scalar'))

            self.metrics.append(MetricInstance(
                metric_identifier,
                metric_fn,
                metric_config_entry.get(reference),
                metric_config_entry.get(threshold),
                metric_presenter
            ))

    def update_metrics_on_object(self, annotation, prediction):
        """
        Updates metric value corresponding given annotation and prediction objects.
        """

        for metric in self.metrics:
            metric.metric_fn.submit(annotation, prediction)

    def update_metrics_on_batch(self, annotation, prediction):
        """
        Updates metric value corresponding given batch.

        Args:
            annotation: list of batch number of annotation objects.
            prediction: list of batch number of prediction objects.
        """

        zipped_transform(self.update_metrics_on_object, annotation, prediction)

    def iterate_metrics(self, annotations, predictions):
        for name, functor, reference, threshold, presenter in self.metrics:
            yield presenter, EvaluationResult(
                name=name,
                evaluated_value=functor(annotations, predictions),
                reference_value=reference,
                threshold=threshold,
                meta=functor.meta,
            )
