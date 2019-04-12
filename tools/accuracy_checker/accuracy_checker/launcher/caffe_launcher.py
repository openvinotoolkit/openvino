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

import caffe

from ..utils import extract_image_representations
from ..config import PathField, StringField, NumberField, BoolField
from .launcher import Launcher, LauncherConfig
from .input_feeder import InputFeeder

DEVICE_REGEX = r'(?P<device>cpu$|gpu)(_(?P<identifier>\d+))?'


class CaffeLauncherConfig(LauncherConfig):
    """
    Specifies configuration structure for Caffe launcher.
    """

    model = PathField()
    weights = PathField()
    device = StringField(regex=DEVICE_REGEX)
    batch = NumberField(floats=False, min_value=1, optional=True)
    output_name = StringField(optional=True)
    allow_reshape_input = BoolField(optional=True)


class CaffeLauncher(Launcher):
    """
    Class for infer model using Caffe framework.
    """

    __provider__ = 'caffe'

    def __init__(self, config_entry: dict, adapter, *args, **kwargs):
        super().__init__(config_entry, adapter, *args, **kwargs)

        caffe_launcher_config = CaffeLauncherConfig('Caffe_Launcher')
        caffe_launcher_config.validate(self._config)

        self.model = str(self._config['model'])
        self.weights = str(self._config['weights'])

        self.network = caffe.Net(self.model, self.weights, caffe.TEST)
        self.allow_reshape_input = self._config.get('allow_reshape_input', False)

        match = re.match(DEVICE_REGEX, self._config['device'].lower())
        if match.group('device') == 'gpu':
            caffe.set_mode_gpu()
            identifier = match.group('identifier') or 0
            caffe.set_device(int(identifier))
        elif match.group('device') == 'cpu':
            caffe.set_mode_cpu()

        self._batch = self._config.get('batch', 1)

        inputs_map = {}
        for input_blob in self.network.inputs:
            inputs_map[input_blob] = self.network.blobs[input_blob]

        self.input_feeder = InputFeeder(self._config.get('inputs') or [], inputs_map)

        if self.adapter:
            self.adapter.output_blob = self.adapter.output_blob or next(iter(self.network.outputs))

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """

        self._inputs_shapes = {}

        for input_blob in self.network.inputs:
            if input_blob in self.input_feeder.const_inputs:
                continue

            channels, height, width = self.network.blobs[input_blob].data.shape[1:]
            self.network.blobs[input_blob].reshape(self._batch, channels, height, width)
            self._inputs_shapes[input_blob] = channels, height, width

        return self._inputs_shapes

    @property
    def batch(self):
        return self._batch

    def predict(self, identifiers, data_representation, *args, **kwargs):
        """
        Args:
            identifiers: list of input data identifiers.
            data_representation: list of input data representations, which contain preprocessed data and its metadata.
        Returns:
            output of model converted to appropriate representation.
        """
        _, meta = extract_image_representations(data_representation)
        dataset_inputs = self.input_feeder.fill_non_constant_inputs(data_representation)
        results = []
        for infer_input in dataset_inputs:
            for input_blob in self.network.inputs:
                if input_blob in self.input_feeder.const_inputs:
                    continue

                data = infer_input[input_blob]

                if self.allow_reshape_input:
                    self.network.blobs[input_blob].reshape(*data.shape)

                if data.shape[0] != self._batch:
                    self.network.blobs[input_blob].reshape(
                        data.shape[0], *self.network.blobs[input_blob].data.shape[1:]
                    )

            results.append(self.network.forward(**self.input_feeder.const_inputs, **infer_input))

        if self.adapter:
            results = self.adapter(results, identifiers, [self._provide_inputs_info_to_meta(meta_) for meta_ in meta])

        return results

    def release(self):
        """
        Releases launcher.
        """

        del self.network
