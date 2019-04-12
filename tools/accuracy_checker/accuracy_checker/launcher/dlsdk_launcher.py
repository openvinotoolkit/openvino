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

import subprocess
from pathlib import Path
import os
import platform
import numpy as np
from cpuinfo import get_cpu_info
import openvino.inference_engine as ie

from ..config import ConfigError, NumberField, PathField, StringField, DictField, ListField, BoolField
from ..logging import warning
from ..utils import read_yaml, contains_all, extract_image_representations, get_path
from .launcher import Launcher, LauncherConfig
from .input_feeder import InputFeeder
from .model_conversion import convert_model
from ..logging import print_info

HETERO_KEYWORD = 'HETERO:'
FPGA_COMPILER_MODE_VAR = 'CL_CONTEXT_COMPILER_MODE_INTELFPGA'
DEVICE_REGEX = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)|(?:^(?P<device>{devices})$)".format(
    hetero=HETERO_KEYWORD, devices="|".join(plugin for plugin in ie.known_plugins)
)


class CPUExtensionPathField(PathField):
    def __init__(self, **kwargs):
        super().__init__(is_directory=False, **kwargs)

    def validate(self, entry, field_uri=None):
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        validation_entry = ''
        try:
            validation_entry = Path(entry)
        except TypeError:
            self.raise_error(entry, field_uri, "values is expected to be path-like")
        is_directory = False
        if validation_entry.parts[-1] == 'AUTO':
            validation_entry = validation_entry.parent
            is_directory = True
        try:
            get_path(validation_entry, is_directory)
        except FileNotFoundError:
            self.raise_error(validation_entry, field_uri, "path does not exist")
        except NotADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is not a directory")
        except IsADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is a directory, regular file expected")


class DLSDKLauncherConfig(LauncherConfig):
    """
    Specifies configuration structure for DLSDK launcher.
    """

    device = StringField(regex=DEVICE_REGEX)
    model = PathField(optional=True)
    weights = PathField(optional=True)
    caffe_model = PathField(optional=True)
    caffe_weights = PathField(optional=True)
    mxnet_weights = PathField(optional=True)
    tf_model = PathField(optional=True)
    onnx_model = PathField(optional=True)
    kaldi_model = PathField(optional=True)
    cpu_extensions = CPUExtensionPathField(optional=True)
    gpu_extensions = PathField(optional=True)
    bitstream = PathField(optional=True)
    mo_params = DictField(optional=True)
    mo_flags = ListField(optional=True)
    outputs = ListField(optional=True)
    allow_reshape_input = BoolField(optional=True)
    affinity_map = PathField(optional=True)
    batch = NumberField(floats=False, min_value=1, optional=True)

    _models_prefix = PathField(is_directory=True, optional=True)
    _model_optimizer = PathField(optional=True, allow_none=True, is_directory=True)
    _tf_obj_detection_api_config_dir = PathField(optional=True, allow_none=True, is_directory=True)
    _tf_custom_op_config_dir = PathField(optional=True, allow_none=True, is_directory=True)
    _cpu_extensions_mode = StringField(optional=True, allow_none=True)
    _aocl = PathField(optional=True)

    def __init__(self, config_uri, **kwargs):
        super().__init__(config_uri, **kwargs)
        self.need_conversion = None

    def validate(self, entry, field_uri=None):
        """
        Validate that launcher entry meets all configuration structure requirements.

        Args:
            entry: launcher configuration file entry.
            field_uri: id of launcher entry.
        """

        dlsdk_model_options = ['model', 'weights']
        caffe_model_options = ['caffe_model', 'caffe_weights']
        mxnet_model_options = ['mxnet_weights']
        tf_model_options = ['tf_model']
        onnx_model_options = ['onnx_model']
        kaldi_model_options = ['kaldi_model']

        multiple_model_sources_err = (
            'Either model and weights or caffe_model and caffe_weights '
            'or mxnet_weights or tf_model should be specified.'
        )
        sources = {
            'dlsdk': dlsdk_model_options,
            'caffe': caffe_model_options,
            'tf': tf_model_options,
            'mxnet': mxnet_model_options,
            'onnx': onnx_model_options,
            'kaldi': kaldi_model_options
        }

        specified = []
        for mo_source_option in sources:
            if contains_all(entry, sources[mo_source_option]):
                specified.append(mo_source_option)

        if not specified:
            raise ConfigError('{} None provided'.format(multiple_model_sources_err))
        if len(specified) > 1:
            raise ConfigError('{} Several provided'.format(multiple_model_sources_err))

        self._set_model_source(specified[0])
        super().validate(entry, field_uri)

    def _set_model_source(self, framework):
        self.need_conversion = framework != 'dlsdk'
        self.framework = framework
        self.fields['model'].optional = self.need_conversion
        self.fields['weights'].optional = self.need_conversion
        self.fields['caffe_model'].optional = framework != 'caffe'
        self.fields['caffe_weights'].optional = framework != 'caffe'
        self.fields['mxnet_weights'].optional = framework != 'mxnet'
        self.fields['tf_model'].optional = framework != 'tf'
        self.fields['onnx_model'].optional = framework != 'onnx'
        self.fields['kaldi_model'].optional = framework != 'kaldi'


class DLSDKLauncher(Launcher):
    """
    Class for infer model using DLSDK framework.
    """

    __provider__ = 'dlsdk'

    def __init__(self, config_entry, adapter):
        super().__init__(config_entry, adapter)

        def fit_to_input(data, input_layer):
            shape_len = len(input_layer.shape)
            if shape_len == 4:
                return np.transpose(data, [0, 3, 1, 2])
            if shape_len == 2:
                if len(np.shape(data)) == 1:
                    return np.transpose([data])
            return np.array(data)

        dlsdk_launcher_config = DLSDKLauncherConfig('DLSDK_Launcher')
        dlsdk_launcher_config.validate(self._config)

        self._device = self._config['device'].upper()
        self._set_variable = False
        self._prepare_bitstream_firmware(self._config)

        if dlsdk_launcher_config.need_conversion:
            self._model, self._weights = DLSDKLauncher.convert_model(self._config, dlsdk_launcher_config.framework)
        else:
            self._model = self._config['model']
            self._weights = self._config['weights']

        self._create_ie_plugin()
        self.network = ie.IENetwork(model=str(self._model), weights=str(self._weights))
        self.original_outputs = self.network.outputs
        outputs = self._config.get('outputs')
        if outputs:
            self.network.add_outputs(outputs)
        self.input_feeder = InputFeeder(
            self._config.get('inputs') or [],
            self.network.inputs,
            prepare_input_data=fit_to_input
        )
        self._batch = self._config.get('batch', self.network.batch_size)
        if self._batch != self.network.batch_size:
            self._set_batch_size(self._batch)
        affinity_map_path = self._config.get('affinity_map')
        if affinity_map_path and self._is_hetero():
            self._set_affinity(affinity_map_path)
        elif affinity_map_path:
            warning('affinity_map config is applicable only for HETERO device')
        self.exec_network = self.plugin.load(network=self.network)
        self.allow_reshape_input = self._config.get('allow_reshape_input', False)

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """

        # reverse and omit N
        return {k: v.shape[1:] for k, v in self.network.inputs.items() if k in self.input_feeder.non_constant_inputs}

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
        _, metadata = extract_image_representations(data_representation)
        non_constant_inputs = self.input_feeder.fill_non_constant_inputs(data_representation)
        results = []
        for infer_inputs in non_constant_inputs:
            input_shapes = {}
            do_reshape = False
            for input_blob in self.network.inputs:
                if input_blob in self.input_feeder.const_inputs:
                    input_shapes[input_blob] = self.network.inputs[input_blob].shape
                    continue

                data = infer_inputs[input_blob]
                input_shapes[input_blob] = data.shape
                if self.allow_reshape_input:
                    if tuple(self.network.inputs[input_blob].shape) != data.shape:
                        do_reshape = True

            if do_reshape:
                self._reshape_input(input_shapes)

            for input_blob, data in infer_inputs.items():
                infer_inputs[input_blob] = self._align_data_shape(data, input_blob)

            network_inputs_data = {**infer_inputs, **self.input_feeder.const_inputs}

            benchmark = kwargs.get('benchmark')
            if benchmark:
                benchmark(network_inputs_data)

            result = self.exec_network.infer(network_inputs_data)

            raw_outputs_callback = kwargs.get('output_callback')
            if raw_outputs_callback:
                raw_outputs_callback(result)

            results.append(result)

        if self.adapter:
            self.adapter.output_blob = self.adapter.output_blob or next(iter(self.original_outputs))
            results = self.adapter(results, identifiers, [self._provide_inputs_info_to_meta(meta) for meta in metadata])

        return results

    def _is_hetero(self):
        return self._device.startswith(HETERO_KEYWORD)

    def _devices_list(self):
        device = self._device
        if HETERO_KEYWORD in self._device:
            device = self._device[len(HETERO_KEYWORD):]

        return [platform_.upper().strip() for platform_ in device.split(',')]

    def _set_affinity(self, affinity_map_path):
        self.plugin.set_initial_affinity(self.network)
        layers = self.network.layers
        for layer, device in read_yaml(affinity_map_path).items():
            if layer not in layers:
                raise ConfigError('Layer \'{layer}\' is not present in network'.format(layer=layer))
            if device not in self._devices_list():
                raise ConfigError(
                    'Device \'{device}\' set for \'{layer}\' layer is not present in '
                    'provided configuration \'{configuration}\''.format(
                        device=device, layer=layer, configuration=self._device
                    )
                )
            layers[layer].affinity = device

    def _is_fpga(self):
        return 'FPGA' in self._devices_list()

    def _prepare_bitstream_firmware(self, config):
        if not self._is_fpga():
            return

        compiler_mode = os.environ.get(FPGA_COMPILER_MODE_VAR)
        if compiler_mode == '3':
            return

        bitstream = config.get('bitstream')
        if bitstream:
            print_info('programming bitstream: {}'.format(bitstream.name))
            aocl_executable = config.get('_aocl')
            if aocl_executable:
                subprocess.run([str(aocl_executable), 'program', 'acl0', str(bitstream)])
                os.environ[FPGA_COMPILER_MODE_VAR] = '3'
                self._set_variable = True
            else:
                aocx_variable = 'DLA_AOCX'
                previous_bitstream = os.environ.get(aocx_variable)
                if previous_bitstream == str(bitstream):
                    return
                os.environ[aocx_variable] = str(bitstream)
                if not os.environ.get(aocx_variable):
                    warning('Warning: {} has not been set'.format(aocx_variable))

    @staticmethod
    def get_cpu_extension(cpu_extensions, selection_mode):
        cpu_extensions_name = cpu_extensions.parts[-1]
        if cpu_extensions_name != 'AUTO':
            return cpu_extensions
        extensions_path = cpu_extensions.parent
        file_format = '{}.dll' if platform.system() == 'Windows' else 'lib{}.so'
        if not selection_mode:
            default_cpu_extension = file_format.format('cpu_extension')
            extension_list = list(extensions_path.glob(default_cpu_extension))

            if extension_list:
                return extension_list[0]

            cpu_info_flags = get_cpu_info()['flags']
            selection_mode = 'avx2' if 'avx2' in cpu_info_flags else 'sse4'
        extension_list = list(extensions_path.glob(file_format.format('cpu_extension_{}'.format(selection_mode))))

        if not extension_list:
            raise ConfigError('suitable CPU extension lib not found in {}'.format(extensions_path))

        return extension_list[0]

    @staticmethod
    def convert_model(config, framework='caffe'):
        config_model = config.get(framework + '_model', '')
        config_weights = config.get(framework + '_weights', '')

        mo_search_paths = []
        model_optimizer = config.get('_model_optimizer')
        if model_optimizer:
            mo_search_paths.append(model_optimizer)

        model_optimizer_directory_env = os.environ.get('MO_DIR')
        if model_optimizer_directory_env:
            mo_search_paths.append(model_optimizer_directory_env)

        return convert_model(
            Path(config_model).name.split('.')[0] or Path(config_weights).name.split('.')[0],
            config_model, config_weights, framework,
            mo_search_paths, config.get('mo_params'),
            config.get('mo_flags'),
            config.get('_tf_custom_op_config_dir'),
            config.get('_tf_obj_detection_api_pipeline_config_path')
        )

    def _reshape_input(self, shapes):
        self.network.reshape(shapes)
        del self.exec_network
        self._create_ie_plugin(log=False)
        self.exec_network = self.plugin.load(network=self.network)

    def _set_batch_size(self, batch_size):
        # in some cases we can not use explicit property for setting batch size, so we need to use reshape instead
        # save const inputs without changes
        const_inputs_shapes = {
            input_name: self.network.inputs[input_name].shape for input_name in self.input_feeder.const_inputs
        }
        new_non_const_input_shapes = {}
        for layer_name in self.input_feeder.non_constant_inputs:
            layer = self.network.inputs[layer_name]
            layer_shape = layer.shape
            ind_batch = layer.layout.find('N')
            if ind_batch != -1:
                layer_shape[ind_batch] = batch_size
            new_non_const_input_shapes[layer_name] = layer_shape

        self.network.reshape({**const_inputs_shapes, **new_non_const_input_shapes})

    def _align_data_shape(self, data, input_blob):
        input_shape = self.network.inputs[input_blob].shape

        if data.shape[0] != input_shape[0]:
            input_shape[0] = data.shape[0]
        if len(data.shape) > 1 and len(input_shape) > 1 and data.shape[1] != input_shape[1]:
            data = data[:, :input_shape[1]]

        return data.reshape(input_shape)

    def _create_ie_plugin(self, log=True):
        if hasattr(self, 'plugin'):
            del self.plugin
        self.plugin = ie.IEPlugin(self._device)
        if log:
            print_info('IE version: {}'.format(ie.get_version()))
            print_info('Loaded {} plugin version: {}'.format(self.plugin.device, self.plugin.version))

        cpu_extensions = self._config.get('cpu_extensions')
        if cpu_extensions and 'CPU' in self._device:
            selection_mode = self._config.get('_cpu_extensions_mode')
            cpu_extensions = DLSDKLauncher.get_cpu_extension(cpu_extensions, selection_mode)
            self.plugin.add_cpu_extension(str(cpu_extensions))
        if self._config.get('gpu_extensions') and 'GPU' in self._device:
            self.plugin.set_config('CONFIG_FILE', str(self._config.get('gpu_extensions')))

    def release(self):
        if self._set_variable:
            del os.environ[FPGA_COMPILER_MODE_VAR]
        del self.network
        del self.exec_network
        del self.plugin
