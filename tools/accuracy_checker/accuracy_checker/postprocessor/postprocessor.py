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

import warnings
from enum import Enum
from ..representation import ContainerRepresentation
from ..config import ConfigValidator, StringField, ConfigError, BaseField
from ..dependency import ClassProvider
from ..utils import (
    zipped_transform,
    string_to_list,
    check_representation_type,
    get_supported_representations,
    enum_values
)


class BasePostprocessorConfig(ConfigValidator):
    type = StringField()
    annotation_source = BaseField(optional=True)
    prediction_source = BaseField(optional=True)


class Postprocessor(ClassProvider):
    __provider_type__ = 'postprocessor'

    annotation_types = ()
    prediction_types = ()

    def __init__(self, config, name=None, meta=None, state=None):
        self.config = config
        self.name = name
        self.meta = meta
        self.state = state
        self.image_size = None

        self.annotation_source = self.config.get('annotation_source')
        if self.annotation_source and not isinstance(self.annotation_source, list):
            self.annotation_source = string_to_list(self.annotation_source)

        self.prediction_source = self.config.get('prediction_source')
        if self.prediction_source and not isinstance(self.prediction_source, list):
            self.prediction_source = string_to_list(self.prediction_source)

        self.validate_config()
        self.setup()

    def __call__(self, *args, **kwargs):
        return self.process_all(*args, **kwargs)

    def setup(self):
        self.configure()

    def process_image(self, annotation, prediction):
        raise NotImplementedError

    def process(self, annotation, prediction):
        image_size = annotation[0].metadata.get('image_size') if not None in annotation else None
        self.image_size = None
        if image_size:
            self.image_size = image_size[0]
        self.process_image(annotation, prediction)

        return annotation, prediction

    def process_all(self, annotations, predictions):
        zipped_transform(self.process, zipped_transform(self.get_entries, annotations, predictions))
        return annotations, predictions

    def configure(self):
        pass

    def validate_config(self):
        BasePostprocessorConfig(
            self.name, on_extra_argument=BasePostprocessorConfig.ERROR_ON_EXTRA_ARGUMENT
        ).validate(self.config)

    def get_entries(self, annotation, prediction):
        message_not_found = '{}: {} is not found in container'
        message_incorrect_type = "Incorrect type of {}. Postprocessor {} can work only with {}"

        def resolve_container(container, supported_types, entry_name, sources=None):
            if not isinstance(container, ContainerRepresentation):
                if sources:
                    message = 'Warning: {}_source can be applied only to container. Default value will be used'
                    warnings.warn(message.format(entry_name))

                return [container]

            if not sources:
                return get_supported_representations(container.values(), supported_types)

            entries = []
            for source in sources:
                representation = container.get(source)
                if not representation:
                    raise ConfigError(message_not_found.format(entry_name, source))

                if supported_types and not check_representation_type(representation, supported_types):
                    raise TypeError(message_incorrect_type.format(entry_name, self.name, ','.join(supported_types)))

                entries.append(representation)

            return entries

        annotation_entries = resolve_container(annotation, self.annotation_types, 'annotation', self.annotation_source)
        prediction_entries = resolve_container(prediction, self.prediction_types, 'prediction', self.prediction_source)

        return annotation_entries, prediction_entries


class ApplyToOption(Enum):
    ANNOTATION = 'annotation'
    PREDICTION = 'prediction'
    ALL = 'all'


class PostprocessorWithTargetsConfigValidator(BasePostprocessorConfig):
    apply_to = StringField(optional=True, choices=enum_values(ApplyToOption))


class PostprocessorWithSpecificTargets(Postprocessor):
    def validate_config(self):
        _config_validator = PostprocessorWithTargetsConfigValidator(
            self.__provider__, on_extra_argument=PostprocessorWithTargetsConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        _config_validator.validate(self.config)

    def setup(self):
        apply_to = self.config.get('apply_to')
        self.apply_to = ApplyToOption(apply_to) if apply_to else None

        if (self.annotation_source or self.prediction_source) and self.apply_to:
            raise ConfigError("apply_to and sources both provided. You need specify only one from them")

        if not self.annotation_source and not self.prediction_source and not self.apply_to:
            raise ConfigError("apply_to or annotation_source or prediction_source required for {}".format(self.name))

        self.configure()

    def process(self, annotation, prediction):
        image_size = annotation[0].metadata.get('image_size') if not None in annotation else None
        self.image_size = None
        if image_size:
            self.image_size = image_size[0]
        target_annotations, target_predictions = None, None
        if self.annotation_source or self.prediction_source:
            target_annotations, target_predictions = self._choose_targets_using_sources(annotation, prediction)

        if self.apply_to:
            target_annotations, target_predictions = self._choose_targets_using_apply_to(annotation, prediction)

        if not target_annotations and not target_predictions:
            raise ValueError("Suitable targets for {} not found".format(self.name))

        self.process_image(target_annotations, target_predictions)
        return annotation, prediction

    def _choose_targets_using_sources(self, annotations, predictions):
        target_annotations = annotations if self.annotation_source else []
        target_predictions = predictions if self.prediction_source else []

        return target_annotations, target_predictions

    def _choose_targets_using_apply_to(self, annotations, predictions):
        targets_specification = {
            ApplyToOption.ANNOTATION: (annotations, []),
            ApplyToOption.PREDICTION: ([], predictions),
            ApplyToOption.ALL: (annotations, predictions)
        }

        return targets_specification[self.apply_to]

    def process_image(self, annotation, prediction):
        raise NotImplementedError
