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

from ..representation import ContainerRepresentation
from ..config import ConfigError
from ..utils import is_single_metric_source, get_supported_representations
from ..presenters import BasePresenter
from ..config import ConfigValidator, NumberField, StringField
from ..dependency import ClassProvider
from ..utils import zipped_transform


class BaseMetricConfig(ConfigValidator):
    type = StringField()
    name = StringField(optional=True)
    reference = NumberField(optional=True)
    threshold = NumberField(min_value=0, optional=True)
    presenter = StringField(choices=BasePresenter.providers, optional=True)
    label_map = StringField(optional=True)
    prediction_source = StringField(optional=True)
    annotation_source = StringField(optional=True)


class Metric(ClassProvider):
    """
    Interface for evaluating metrics.
    """

    __provider_type__ = 'metric'

    annotation_types = ()
    prediction_types = ()

    _config_validator_type = BaseMetricConfig

    def __init__(self, config, dataset, name=None, state=None):
        self.config = config
        self.name = name
        self.dataset = dataset
        self.state = state
        self._update_iter = 0
        self.meta = {}

        self.validate_config()
        self.configure()
        message_unsupported_multi_source = 'metric {} does not support several {} sources'
        self.annotation_source = self.config.get('annotation_source')

        if self.annotation_source and not is_single_metric_source(self.annotation_source):
            raise ConfigError(message_unsupported_multi_source.format(self.name, 'annotation'))

        self.prediction_source = self.config.get('prediction_source')
        if self.prediction_source and not is_single_metric_source(self.prediction_source):
            raise ConfigError(message_unsupported_multi_source.format(self.name, 'prediction'))

    def __call__(self, *args, **kwargs):
        return self.submit_all(*args, **kwargs)

    def submit(self, annotation, prediction):
        self.update(annotation, prediction)

    def submit_all(self, annotations, predictions):
        return self.evaluate(annotations, predictions)

    def update(self, annotation, prediction):
        pass

    def evaluate(self, annotations, predictions):
        raise NotImplementedError

    def configure(self):
        """
        Specifies configuration structure for metric entry.
        """

        pass

    def validate_config(self):
        """
        Validate that metric entry meets all configuration structure requirements.
        """

        self._config_validator_type(
            self.name, on_extra_argument=BaseMetricConfig.ERROR_ON_EXTRA_ARGUMENT
        ).validate(self.config)

    def _update_state(self, fn, state_key, default_factory=None):
        iter_key = "{}_global_it".format(state_key)
        if state_key not in self.state:
            default = default_factory() if default_factory else None
            self.state[state_key] = default
            self.state[iter_key] = 0

        self._update_iter += 1
        if self.state[iter_key] < self._update_iter:
            self.state[iter_key] += 1
            self.state[state_key] = fn(self.state[state_key])

    def _resolve_representation_containers(self, annotation, prediction):
        def get_resolve_subject(representation, source=None):
            def is_container(representation):
                if isinstance(representation, ContainerRepresentation):
                    return True
                representation_parents = type(representation).__bases__
                representation_parents_names = [parent.__name__ for parent in representation_parents]

                return ContainerRepresentation.__name__ in representation_parents_names

            if not is_container(representation):
                return representation

            if not source:
                return representation.values()

            representation = representation.get(source)
            if not representation:
                raise ConfigError('{} not found'.format(source))

            return representation

        annotation = get_resolve_subject(annotation, self.annotation_source)
        prediction = get_resolve_subject(prediction, self.prediction_source)

        def resolve(representation, supported_types, representation_name):
            message_not_found = 'suitable {} for metric {} not found'
            message_need_source = 'you need specify {} source for metric {}'

            representation = get_supported_representations(representation, supported_types)
            if not representation:
                raise ConfigError(message_not_found.format(representation_name, self.name))

            if len(representation) > 1:
                raise ConfigError(message_need_source.format(representation_name, self.name))

            return representation[0]

        resolved_annotation = resolve(annotation, self.annotation_types, 'annotation')
        resolved_prediction = resolve(prediction, self.prediction_types, 'prediction')

        return resolved_annotation, resolved_prediction


class PerImageEvaluationMetric(Metric):
    def submit(self, annotation, prediction):
        annotation_, prediction_ = self._resolve_representation_containers(annotation, prediction)
        self.update(annotation_, prediction_)

    def evaluate(self, annotations, predictions):
        raise NotImplementedError


class FullDatasetEvaluationMetric(Metric):
    def submit_all(self, annotations, predictions):
        annotations_, predictions_ = zipped_transform(self._resolve_representation_containers, annotations, predictions)
        return self.evaluate(annotations_, predictions_)

    def evaluate(self, annotations, predictions):
        raise NotImplementedError
