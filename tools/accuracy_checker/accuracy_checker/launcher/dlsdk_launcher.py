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
from ..utils import read_yaml, contains_all, get_path, contains_any
from .launcher import Launcher, LauncherConfig
from .model_conversion import convert_model, FrameworkParameters
from ..logging import print_info

HETERO_KEYWORD = 'HETERO:'
FPGA_COMPILER_MODE_VAR = 'CL_CONTEXT_COMPILER_MODE_INTELFPGA'
DEVICE_REGEX = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)|(?:^(?P<device>{devices})$)".format(
    hetero=HETERO_KEYWORD, devices="|".join(plugin for plugin in ie.known_plugins)
)
VPU_PLUGINS = ('HDDL', "MYRIAD")
VPU_LOG_LEVELS = ('LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG')


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
    tf_meta = PathField(optional=True)
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
    _vpu_log_level = StringField(optional=True, choices=VPU_LOG_LEVELS)

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
        tf_meta_options = ['tf_meta']
        onnx_model_options = ['onnx_model']
        kaldi_model_options = ['kaldi_model']

        multiple_model_sources_err = (
            'Either model and weights or caffe_model and caffe_weights '
            'or mxnet_weights or tf_model or tf_meta should be specified.'
        )
        sources = {
            FrameworkParameters('dlsdk', False): dlsdk_model_options,
            FrameworkParameters('caffe', False): caffe_model_options,
            FrameworkParameters('tf', False): tf_model_options,
            FrameworkParameters('mxnet', False): mxnet_model_options,
            FrameworkParameters('onnx', False): onnx_model_options,
            FrameworkParameters('kaldi', False): kaldi_model_options,
            FrameworkParameters('tf', True): tf_meta_options
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
        self.need_conversion = framework.name != 'dlsdk'
        self.framework = framework
        self.fields['model'].optional = self.need_conversion
        self.fields['weights'].optional = self.need_conversion
        self.fields['caffe_model'].optional = framework.name != 'caffe'
        self.fields['caffe_weights'].optional = framework.name != 'caffe'
        self.fields['mxnet_weights'].optional = framework.name != 'mxnet'
        self.fields['tf_model'].optional = framework != FrameworkParameters('tf', False)
        self.fields['tf_meta'].optional = framework != FrameworkParameters('tf', True)
        self.fields['onnx_model'].optional = framework.name != 'onnx'
        self.fields['kaldi_model'].optional = framework.name != 'kaldi'


class DLSDKLauncher(Launcher):
    """
    Class for infer model using DLSDK framework.
    """

    __provider__ = 'dlsdk'

    def __init__(self, config_entry):
        super().__init__(config_entry)

        dlsdk_launcher_config = DLSDKLauncherConfig('DLSDK_Launcher')
        dlsdk_launcher_config.validate(self.config)

        self._device = self.config['device'].upper()
        self._set_variable = False
        self._prepare_bitstream_firmware(self.config)

        if dlsdk_launcher_config.need_conversion:
            self._model, self._weights = DLSDKLauncher.convert_model(self.config, dlsdk_launcher_config.framework)
        else:
            self._model = self.config['model']
            self._weights = self.config['weights']

        self._create_ie_plugin()
        self.network = ie.IENetwork(model=str(self._model), weights=str(self._weights))
        self.original_outputs = self.network.outputs
        outputs = self.config.get('outputs')
        if outputs:
            self.network.add_outputs(outputs)
        self.const_inputs = self.config.get('_list_const_inputs', [])
        self._batch = self.config.get('batch', self.network.batch_size)
        if self._batch != self.network.batch_size:
            self._set_batch_size(self._batch)
        affinity_map_path = self.config.get('affinity_map')
        if affinity_map_path and self._is_hetero():
            self._set_affinity(affinity_map_path)
        elif affinity_map_path:
            warning('affinity_map config is applicable only for HETERO device')
        self.exec_network = self.plugin.load(network=self.network)
        self.allow_reshape_input = self.config.get('allow_reshape_input', False)

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """

        # reverse and omit N
        return {k: v.shape[1:] for k, v in self.network.inputs.items() if k not in self.const_inputs}

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self.original_outputs))

    def predict(self, inputs, metadata, *args, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_inputs in inputs:
            input_shapes = {}
            do_reshape = False
            for input_blob in self.network.inputs:
                if input_blob in self.const_inputs:
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
                if input_blob in self.const_inputs:
                    continue
                infer_inputs[input_blob] = self._align_data_shape(data, input_blob)

            network_inputs_data = {**infer_inputs}

            benchmark = kwargs.get('benchmark')
            if benchmark:
                benchmark(network_inputs_data)

            result = self.exec_network.infer(network_inputs_data)

            raw_outputs_callback = kwargs.get('output_callback')
            if raw_outputs_callback:
                raw_outputs_callback(result)

            results.append(result)
            for meta_ in metadata:
                self._provide_inputs_info_to_meta(meta_)

            for meta in metadata:
                self._provide_inputs_info_to_meta(meta)

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

    def _is_vpu(self):
        return contains_any(self._devices_list(), VPU_PLUGINS)

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
                subprocess.run([str(aocl_executable), 'program', 'acl0', str(bitstream)], check=True)
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
            supported_flags = ['avx512', 'avx2', 'sse4']
            for flag in supported_flags:
                selection_mode = flag
                if selection_mode in cpu_info_flags:
                    break
        extension_list = list(extensions_path.glob(file_format.format('cpu_extension_{}'.format(selection_mode))))

        if not extension_list:
            raise ConfigError('suitable CPU extension lib not found in {}'.format(extensions_path))

        return extension_list[0]

    @staticmethod
    def convert_model(config, framework=FrameworkParameters('caffe', False)):
        config_model = config.get('{}_model'.format(framework.name), '')
        config_weights = config.get('{}_weights'.format(framework.name), '')
        config_meta = config.get('{}_meta'.format(framework.name), '')

        mo_search_paths = []
        model_optimizer = config.get('_model_optimizer')
        if model_optimizer:
            mo_search_paths.append(model_optimizer)

        model_optimizer_directory_env = os.environ.get('MO_DIR')
        if model_optimizer_directory_env:
            mo_search_paths.append(model_optimizer_directory_env)

        model_name = (
            Path(config_model).name.rsplit('.', 1)[0] or
            Path(config_weights).name.rsplit('.', 1)[0] or
            Path(config_meta).name.rsplit('.', 1)[0]
        )

        return convert_model(
            model_name,
            config_model, config_weights, config_meta, framework,
            mo_search_paths, config.get('mo_params'),
            config.get('mo_flags'),
            config.get('_tf_custom_op_config_dir'),
            config.get('_tf_obj_detection_api_pipeline_config_path')
        )

    def get_all_inputs(self):
        return self.network.inputs

    def _reshape_input(self, shapes):
        self.network.reshape(shapes)
        del self.exec_network
        self.exec_network = self.plugin.load(network=self.network)

    def _set_batch_size(self, batch_size):
        # in some cases we can not use explicit property for setting batch size, so we need to use reshape instead
        # save const inputs without changes
        const_inputs_shapes = {
            input_name: self.network.inputs[input_name].shape for input_name in self.const_inputs
        }
        new_non_const_input_shapes = {}
        for layer_name, layer in self.network.inputs.items():
            if layer_name in const_inputs_shapes:
                continue
            layer_shape = layer.shape
            ind_batch = layer.layout.find('N')
            if ind_batch != -1:
                layer_shape[ind_batch] = batch_size
            new_non_const_input_shapes[layer_name] = layer_shape

        self.network.reshape({**const_inputs_shapes, **new_non_const_input_shapes})

    def _align_data_shape(self, data, input_blob):
        input_shape = self.network.inputs[input_blob].shape
        data_batch_size = data.shape[0]
        input_batch_size = input_shape[0]

        if data_batch_size < input_batch_size:
            warning_message = 'data batch {} is not equal model input batch_size {}. '.format(
                data_batch_size, input_batch_size
            )
            warning(warning_message)
            diff_number = input_batch_size - data_batch_size
            filled_part = [data[-1]] * diff_number
            data = np.concatenate([data, filled_part])

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

        cpu_extensions = self.config.get('cpu_extensions')
        if cpu_extensions and 'CPU' in self._devices_list():
            selection_mode = self.config.get('_cpu_extensions_mode')
            cpu_extensions = DLSDKLauncher.get_cpu_extension(cpu_extensions, selection_mode)
            self.plugin.add_cpu_extension(str(cpu_extensions))
        gpu_extensions = self.config.get('gpu_extensions')
        if gpu_extensions and 'GPU' in self._devices_list():
            self.plugin.set_config('CONFIG_FILE', str(gpu_extensions))
        if self._is_vpu():
            log_level = self.config.get('_vpu_log_level')
            if log_level:
                self.plugin.set_config({'VPU_LOG_LEVEL': log_level})

    @staticmethod
    def fit_to_input(data, input_layer):
        shape_len = len(input_layer.shape)
        if shape_len == 4:
            if len(np.shape(data)) == 5:
                data = data[0]
            return np.transpose(data, [0, 3, 1, 2])
        if shape_len == 2:
            if len(np.shape(data)) == 1:
                return np.transpose([data])
        return np.array(data)

    def release(self):
        if 'network' in self.__dict__:
            del self.network
        if 'exec_network' in self.__dict__:
            del self.exec_network
        if 'plugin' in self.__dict__:
            del self.plugin
        if self._set_variable:
            del os.environ[FPGA_COMPILER_MODE_VAR]
