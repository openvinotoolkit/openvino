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
from ..config import BaseField
from ..adapters import AdapterField
from ..config import ConfigValidator, StringField, ListField
from ..dependency import ClassProvider


class Launcher(ClassProvider):
    """
    Interface for inferring model.
    """

    __provider_type__ = 'launcher'

    def __init__(self, config_entry, *args, **kwargs):
        self.config = config_entry

    def predict(self, inputs, metadata, *args, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """

        raise NotImplementedError

    def __call__(self, context, *args, **kwargs):
        context.prediction_batch = self.predict(context.input_blobs, context.batch_meta)


    def get_all_inputs(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    @property
    def batch(self):
        raise NotImplementedError

    @property
    def output_blob(self):
        raise NotImplementedError

    @property
    def inputs(self):
        raise NotImplementedError

    def _provide_inputs_info_to_meta(self, meta):
        meta['input_shape'] = self.inputs

        return meta

    @staticmethod
    def fit_to_input(data, input_layer):
        if len(np.shape(data)) == 4:
            return np.transpose(data, [0, 3, 1, 2])
        return np.array(data)

INPUTS_TYPES = ('CONST_INPUT', 'INPUT')

class InputValidator(ConfigValidator):
    name = StringField()
    type = StringField(choices=INPUTS_TYPES)
    value = BaseField()


class ListInputsField(ListField):
    def __init__(self, **kwargs):
        super().__init__(allow_empty=False, value_type=InputValidator('Inputs'), **kwargs)

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        names_set = set()
        for input_layer in entry:
            input_name = input_layer['name']
            if input_name not in names_set:
                names_set.add(input_name)
            else:
                self.raise_error(entry, field_uri, '{} repeated name'.format(input_name))


class LauncherConfig(ConfigValidator):
    """
    Specifies common part of configuration structure for launchers.
    """

    framework = StringField(choices=Launcher.providers)
    tags = ListField(allow_empty=False, optional=True)
    inputs = ListInputsField(optional=True)
    adapter = AdapterField(optional=True)

    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        inputs = entry.get('inputs')
        if inputs:
            inputs_by_type = {input_type: [] for input_type in INPUTS_TYPES}
            for input_layer in inputs:
                input_type = input_layer['type']
                inputs_by_type[input_type].append(input_layer['name'])

            additional_attributes = {
                '_list_{}s'.format(input_type.lower()): inputs for input_type, inputs in inputs_by_type.items()
            }
            for additional_attribute, values in additional_attributes.items():
                self.fields[additional_attribute] = values


def unsupported_launcher(name, error_message=None):
    class UnsupportedLauncher(Launcher):
        __provider__ = name

        def __init__(self, config_entry, *args, **kwargs):
            super().__init__(config_entry, *args, **kwargs)

            msg = "{launcher} launcher is disabled. Please install {launcher} to enable it.".format(launcher=name)
            raise ValueError(error_message or msg)

        def predict(self, identifiers, data, *args, **kwargs):
            raise NotImplementedError

        def release(self):
            raise NotImplementedError

        @property
        def batch(self):
            raise NotImplementedError

    return UnsupportedLauncher


def create_launcher(launcher_config):
    """
    Args:
        launcher_config: launcher configuration file entry.
    Returns:
        framework-specific launcher object.
    """

    launcher_config_validator = LauncherConfig(
        'Launcher_validator',
        on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
    )
    launcher_config_validator.validate(launcher_config)
    config_framework = launcher_config['framework']

    return Launcher.provide(config_framework, launcher_config)
