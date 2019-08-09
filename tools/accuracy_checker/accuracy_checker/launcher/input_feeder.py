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

import re
import numpy as np

from ..config import ConfigError
from ..utils import extract_image_representations


class InputFeeder:
    def __init__(self, inputs_config, network_inputs, prepare_input_data=None):
        def fit_to_input(data, input_layer):
            if len(np.shape(data)) == 4:
                return np.transpose(data, [0, 3, 1, 2])
            return np.array(data)

        self.input_transform_func = prepare_input_data or fit_to_input
        self.network_inputs = network_inputs
        self.configure(inputs_config)

    def configure(self, inputs_config):
        self.const_inputs, self.non_constant_inputs, self.inputs_mapping = self._parse_inputs_config(inputs_config)
        if not self.non_constant_inputs:
            raise ConfigError('Network should contain at least one layer for setting variable data.')

    def fill_non_constant_inputs(self, data_representation_batch):
        filled_inputs = {}
        for input_layer in self.non_constant_inputs:
            input_regex = None
            input_batch = []
            if self.inputs_mapping:
                input_regex = self.inputs_mapping[input_layer]
            for data_representation in data_representation_batch:
                input_data = None
                identifiers = data_representation.identifier
                data = data_representation.data
                if not isinstance(identifiers, list) and not input_regex:
                    input_data = data
                    input_batch.append(input_data)
                    continue

                if not input_regex:
                    raise ConfigError('Impossible to choose correct data for layer {}.'
                                      'Please provide regular expression for matching in config.'.format(input_layer))
                data = [data] if np.isscalar(identifiers) else data
                identifiers = [identifiers] if np.isscalar(identifiers) else identifiers
                for identifier, data_value in zip(identifiers, data):
                    if input_regex.match(identifier):
                        input_data = data_value
                        break
                if input_data is None:
                    raise ConfigError('Suitable data for filling layer {} not found'.format(input_layer))
                input_batch.append(input_data)

            filled_inputs[input_layer] = input_batch

        return self._transform_batch(filled_inputs, extract_image_representations(data_representation_batch)[1])

    def fill_inputs(self, data_representation_batch):
        inputs = self.fill_non_constant_inputs(data_representation_batch)
        for infer_inputs in inputs:
            infer_inputs.update(self.const_inputs)
        return inputs

    def __call__(self, context, *args, **kwargs):
        data_batch = context.data_batch
        _, meta = extract_image_representations(data_batch)
        context.input_blobs = self.fill_inputs(data_batch)
        context.batch_meta = meta

    def _parse_inputs_config(self, inputs_entry):
        constant_inputs = {}
        non_constant_inputs_mapping = {}
        non_constant_inputs = []
        for input_ in inputs_entry:
            name = input_['name']
            if not name in self.network_inputs:
                raise ConfigError('network does not contain input "{}"'.format(name))
            value = input_['value']

            if input_['type'] == 'CONST_INPUT':
                if isinstance(value, list):
                    value = np.array(value)
                constant_inputs[name] = value
            else:
                value = re.compile(value)
                non_constant_inputs_mapping[name] = value

        non_constant_inputs = list(non_constant_inputs_mapping.keys())
        not_config_inputs = list(filter(
            lambda input_layer: not input_layer in non_constant_inputs + list(constant_inputs.keys()),
            self.network_inputs.keys()
            ))
        if non_constant_inputs and not_config_inputs:
            raise ConfigError('input value for {} are not presented in config.'.format(','.join(not_config_inputs)))
        non_constant_inputs += not_config_inputs

        return constant_inputs, non_constant_inputs, non_constant_inputs_mapping or None

    def _transform_batch(self, batch_data, meta):
        def calculate_num_splits(layers_data, batch_size):
            max_split_num = 1
            for _, data in layers_data.items():
                total_tiles_num = 0
                for tiles in data:
                    total_tiles_num += len(tiles)

                offset = 0 if total_tiles_num % batch_size == 0 else 1
                splits_for_layer = (total_tiles_num // batch_size) + offset
                if max_split_num < splits_for_layer:
                    max_split_num = splits_for_layer

            return max_split_num

        def separate_data(data, num_splits):
            grouped_data = [[] for _ in range(num_splits)]
            for data_part in data:
                for split_id, data_split in enumerate(data_part):
                    grouped_data[split_id % num_splits].append(data_split)
            return grouped_data

        batch_size = len(meta)
        if meta[-1].get('multi_infer', False):
            num_splits = calculate_num_splits(batch_data, batch_size)
            infers_data = [{} for _ in range(num_splits)]
            for layer_name, layer_data in batch_data.items():
                batch_for_all_infers = separate_data(layer_data, num_splits)
                for infer_id, on_infer_batch in enumerate(batch_for_all_infers):
                    infers_data[infer_id][layer_name] = self.input_transform_func(
                        on_infer_batch, self.network_inputs[layer_name]
                    )
            return infers_data

        for layer_name, layer_data in batch_data.items():
            batch_data[layer_name] = self.input_transform_func(layer_data, self.network_inputs[layer_name])

        return [batch_data]
