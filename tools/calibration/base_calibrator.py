"""
Copyright (C) 2018-2019 Intel Corporation

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

from abc import abstractmethod
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Dict

import openvino.inference_engine as ie

from accuracy_checker.progress_reporters import TQDMReporter, ProgressReporter
from accuracy_checker.evaluators.model_evaluator import ModelEvaluator
from accuracy_checker.presenters import get_result_format_parameters

from ..utils.network_info import NetworkInfo
from ..utils.building.network_builder import NetworkBuilder
from ..utils.building.layer import Layer

from .logging import info, debug
from .calibrator_configuration import CalibratorConfiguration
from .nrmsd import compare_nrmsd
from .single_layer_network import SingleLayerNetwork
from .inference_result import InferenceResult
from .calibration_metrics import CalibrationMetrics
from .accuracy.metric_factory import MetricFactory

from .process_dataset_callbacks.collect_results_callback import CollectResultsCallback
from .process_dataset_callbacks.calculate_accuracy_callback import CalculateAccuracyCallback
from ..utils.weights import Weights
from ..utils.biases import Biases

class MetricsCallback:
    def __init__(self):
        self._values = list()
        self._latencies = list()

    def callback(self, value, latency = None):
        self._values.append(value)
        self._latencies.append(latency)

    @property
    def values(self):
        return self._values

    @property
    def latencies(self):
        return self._latencies

class BaseCalibrator:
    '''
    Base type for all calibrators
    '''

    def __init__(self, configuration: CalibratorConfiguration):
        self._configuration = configuration

        network = self.create_network()
        self._input_layer_name = next(iter(network.inputs))
        self._output_layer_name = next(iter(network.outputs))

        self.plugin = ie.IEPlugin(self._configuration.device)
        if self._configuration.cpu_extension and self._configuration.device == 'CPU':
            self.plugin.add_cpu_extension(self._configuration.cpu_extension)
        if self._configuration.gpu_extension and self._configuration.device == 'GPU':
            self.plugin.set_config('CONFIG_FILE', self._configuration.gpu_extension)

    def get_allowed_outputs(self, desired_layers: list=None) -> list:
        network_tmp = self.create_network()
        # During network loading some layers are trancated. Outputs could not be added to these layers
        self.plugin.load(network_tmp)

        output_names = list()
        excluded_list = ['gather']
        children_require_stat = ['convolution', 'fullyconnected']
        for layer in network_tmp.layers.values():
            add = False
            for child_name in layer.children:
                if network_tmp.layers[child_name].type.lower() not in excluded_list:
                    add = True
                    break
            if layer.type.lower() == "gather":
                add = False
                for child_name in layer.children:
                    if network_tmp.layers[child_name].type.lower() in children_require_stat:
                        add = True
                        break
            if add:
                output_names.append(layer.name)

        # return just custom layers if they were set
        if desired_layers:
            custom_layers_list = list()
            for name in output_names:
                if name in desired_layers:
                    custom_layers_list.append(name)
            return custom_layers_list

        return output_names

    def create_network(self) -> ie.IENetwork:
        network = ie.IENetwork(self._configuration.model, self._configuration.weights)
        if len(network.outputs) == 0:
            raise ValueError("no outputs")
        if len(network.inputs) == 0:
            raise ValueError("no inputs")
        return network

    def create_network_for_layer(
        self,
        weights: str,
        quantization_layer: ie.IENetLayer,
        quantization_layer_info: Layer,
        activation_layer: ie.IENetLayer):

        quantization_layer_weights = None
        quantization_layer_biases = None
        weights_file_path_has_to_be_removed = False
        try:
            if weights is None:
                weights_file_path_has_to_be_removed = True
                weights_file_path = os.path.join(tempfile._get_default_tempdir(), "model_" + next(tempfile._get_candidate_names()) + ".bin")
                with open(weights_file_path, 'wb') as bin_file:
                    for blob_name, blob_content in quantization_layer.weights.items():
                        start = bin_file.tell()
                        blob_content.tofile(bin_file)
                        end = bin_file.tell()
                        if blob_name == 'weights':
                            quantization_layer_weights = Weights(start, end - start)
                        if blob_name == 'biases':
                            quantization_layer_biases = Biases(start, end - start)

                if quantization_layer_weights is None:
                    raise ValueError("quantization layer '{}' doesn't contain weights".format(quantization_layer.name))
            else:
                weights_file_path = weights
                quantization_layer_weights = quantization_layer_info.weights
                quantization_layer_biases = quantization_layer_info.biases

            if self.is_quantization_supported(quantization_layer.type):
                input_layer_info = quantization_layer_info.inputs[0].layer

                layers = [
                    Layer(
                        0,
                        "Input",
                        input_layer_info.name,
                        quantization_layer.precision,
                        {},
                        [],
                        input_layer_info.outputs[0].port.dim),

                    Layer(
                        1,
                        quantization_layer.type,
                        quantization_layer.name,
                        quantization_layer.precision,
                        quantization_layer.params,
                        quantization_layer_info.inputs[0].port.dim,
                        quantization_layer_info.outputs[0].port.dim,
                        quantization_layer_weights,
                        quantization_layer_biases)
                ]

                if activation_layer:
                    activation_layer_info = quantization_layer_info.outputs[0].layer
                    reference_output_layer_name = activation_layer_info.name
                    outputs = activation_layer_info.outputs
                    output_layer_outputs_dim = \
                        outputs[0].port.dim if outputs else activation_layer_info.inputs[0].port.dim

                    layers.append(Layer(
                        len(layers),
                        activation_layer.type,
                        activation_layer.name,
                        activation_layer.precision,
                        activation_layer.params,
                        activation_layer_info.inputs[0].port.dim,
                        output_layer_outputs_dim))
                else:
                    reference_output_layer_name = quantization_layer_info.name
                    output_layer_outputs_dim = quantization_layer_info.outputs[0].port.dim

                layers.append(Layer(
                    len(layers),
                    "Power",
                    quantization_layer.name + "_",
                    quantization_layer.precision,
                    {'power': 1.0, 'scale': 1.0, 'shift': 0.0},
                    output_layer_outputs_dim,
                    output_layer_outputs_dim))

                builder = NetworkBuilder().sequential(layers)
            else:
                raise ValueError("unsupported layer type '{}'".format(quantization_layer.type))

            # filling weights and biases

            temporary_file = tempfile.NamedTemporaryFile(delete=False)
            try:
                builder_str = str(builder)
                network_content = str.encode(builder_str)
                temporary_file.write(network_content)
                temporary_file.close()

                network_for_layer_model = temporary_file.name
                network_for_layer = ie.IENetwork(network_for_layer_model, weights_file_path)
                network_for_layer.add_outputs([quantization_layer.name + "_"])
            finally:
                if os.path.exists(temporary_file.name):
                    temporary_file.close()
                    os.remove(temporary_file.name)
        finally:
            if weights_file_path_has_to_be_removed and os.path.exists(weights_file_path):
                os.remove(weights_file_path)

        return network_for_layer, reference_output_layer_name

    def save(self, model_file_path: str, weights_file_path: str, quantization_level: dict, statistics):
        '''
        Save calibration results.
        '''


        if not statistics:
            raise ValueError("statistics is empy")

        network = self.create_network()

        network_stats = {}
        for layer_name, node_statistic in statistics.items():
            network_stats[layer_name] = ie.LayerStats(min=tuple(node_statistic.min_outputs),
                                                      max=tuple(node_statistic.max_outputs))
        network.stats.update(network_stats)

        for layer in network.layers.values():
            if self.is_quantization_supported(layer.type) and layer.name in quantization_level:
                params = layer.params
                params["quantization_level"] = quantization_level[layer.name]
                layer.params = params

        network.serialize(model_file_path, weights_file_path)

    @staticmethod
    def __parse_inputs(inputs_entry):
        inputs = {}
        for input_ in inputs_entry:
            value = input_['value']
            if isinstance(value, list):
                value = np.array(value)

            inputs[input_['name']] = value

        return inputs

    @staticmethod
    def compare_result(result1, result2, output_name: str):
        if len(result1) != len(result2):
            return False

        for index in range(len(result1)):
            result_map1 = result1[index]
            result_map2 = result2[index]

            compare_result = result_map1[output_name] == result_map2[output_name]
            if not compare_result.all():
                debug('\nresult_map1={}\n'.format(result_map1[output_name]))
                debug('\nresult_map2={}\n'.format(result_map2[output_name]))
                return False
        return True

    def infer_single_layer_network(self,
                                   single_layer_network: SingleLayerNetwork,
                                   full_network_result: InferenceResult):
        '''
        Native infer and compare results
        '''

        if single_layer_network.input_layer_name in full_network_result:
            input_layer_data = full_network_result[single_layer_network.input_layer_name]
        else:
            raise ValueError("single layer network input '{}' was not found in reference inference".format(
                single_layer_network.input_layer_name))

        single_layer_network_result = \
            single_layer_network.exec_network.infer({single_layer_network.input_layer_name: input_layer_data})
        if single_layer_network.output_layer_name not in single_layer_network_result:
            raise ValueError("singld layer network output layer '{}' was not found in single"
                             " layer inference result".format(single_layer_network.layer_name))
        actual_result_data = single_layer_network_result[single_layer_network.output_layer_name]

        if single_layer_network.reference_output_layer_name not in full_network_result:
            raise ValueError("single layer network output layer '{}' was not found in "
                             "full inference result".format(single_layer_network.layer_name))
        expected_result_data = full_network_result[single_layer_network.reference_output_layer_name]

        accuracy_drop = compare_nrmsd(actual_result_data, expected_result_data)
        return accuracy_drop

    def infer(
        self,
        model_path=None,
        collect_aggregated_statistics: bool = False,
        collect_resuls: bool = False,
        collect_layers: set = None,
        collect_performance_counters: bool = False,
        ignore_layer_names: list = None,
        per_layer_statistics: dict = None,
        add_outputs=False
    ) -> InferenceResult:
        '''
        Accuracy checker infer and compare results
        '''
        accuracy = 0.0

        model = self._configuration.config['models'][0]
        # Need to copy to keep origin configuration here
        launcher_config = model['launchers'][0].copy()
        dataset_config = model['datasets'][0].copy()

        if model_path:
            launcher_config['model'] = Path(model_path)
            launcher_config['weights'] = Path(model_path[:len(model_path) - 3] + 'bin')

        if add_outputs:
            launcher_config['outputs'] = self.get_allowed_outputs()

        process_dataset_callback = None
        model_evaluator = ModelEvaluator.from_configs(launcher_config, dataset_config)
        try:
            if collect_performance_counters:
                model_evaluator.launcher.plugin.set_config({'PERF_COUNT': 'YES'})

            dataset_size = model_evaluator.dataset.size

            if self._configuration.progress:
                progress_reporter = ProgressReporter.provide((
                    self._configuration.progress if ':' not in self._configuration.progress
                    else self._configuration.progress.split(':')[0]
                ))
                progress_reporter.reset(len(model_evaluator.dataset))
            else :
                progress_reporter = None

            if collect_layers:
                process_dataset_callback = CalculateAccuracyCallback(
                    model_evaluator.launcher.network,
                    model_evaluator.launcher.exec_network,
                    collect_layers=collect_layers,
                    configuration=self._configuration,
                    per_layer_statistics=per_layer_statistics,
                    normalizer=self,
                    ignore_layer_names=ignore_layer_names)
            else:
                process_dataset_callback = CollectResultsCallback(
                    model_evaluator.launcher.network,
                    model_evaluator.launcher.exec_network,
                    collect_resuls=collect_resuls,
                    collect_layers=collect_layers,
                    collect_aggregated_statistics=collect_aggregated_statistics,
                    iterations_count=int(dataset_size / self._configuration.batch_size),
                    dataset_size=dataset_size)


            model_evaluator.process_dataset(None,
                                            progress_reporter=progress_reporter,
                                            output_callback=process_dataset_callback.callback)
            if len(model_evaluator.launcher.exec_network.requests) != 1:
                raise ValueError("unexpected network requests count")

            inference_result = process_dataset_callback.infer_raw_result
            layers_accuracy_drop = process_dataset_callback.get_accuracy_drop() if collect_layers else None
            inference_latencies = process_dataset_callback.latencies

            performance_counters = \
                model_evaluator.launcher.exec_network.requests[0].get_perf_counts() if collect_performance_counters else None

            model_evaluator_callback = MetricsCallback()
            model_evaluator.compute_metrics(output_callback=model_evaluator_callback.callback)
            presenter_values = model_evaluator_callback.values
            postfix = ""
            scale = ""
            for presenter_value in presenter_values:
                value, reference, name, metric_type, threshold, meta = presenter_value
                info("Accuracy checker meta data: '{}'".format(meta))
                postfix, scale, result_format = get_result_format_parameters(meta, False)
                if np.isscalar(value):
                    accuracy = value
                # In case of vectorized metric with one number 'value'l could be array with zero shape
                elif meta.get('calculate_mean', True) or (isinstance(value, np.ndarray) and len(value.shape) == 0):
                    info("'mean' value is used as accuracy")
                    accuracy = np.mean(value)
                elif len(value) > 0:
                    accuracy = value[0]
                else:
                    raise Exception('Unsupported model evaluation output')
        except Exception:

            if process_dataset_callback:
                process_dataset_callback.release()
            raise
        finally:
            model_evaluator.release()

        return InferenceResult(
            inference_result,
            CalibrationMetrics(MetricFactory.create(scale, postfix, accuracy), np.mean(inference_latencies))
                if len(inference_latencies)
                else CalibrationMetrics(MetricFactory.create(scale, postfix, accuracy)),
            aggregated_statistics=process_dataset_callback.aggregated_statistics,
            performance_counters=performance_counters,
            layers_accuracy_drop=layers_accuracy_drop)

    def get_quantization_levels(self, ignore_layer_names=None) -> Dict[str, str]:
        network = self.create_network()
        quantization_levels = dict()

        for layer in network.layers.values():
            if self.is_quantization_supported(layer.type):
                if ignore_layer_names and (layer.name in ignore_layer_names):
                    quantization_levels[layer.name] = layer.precision
                else:
                    quantization_levels[layer.name] = "I8" if self.precision == "INT8" else self.precision

        return quantization_levels

    @property
    def precision(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def is_quantization_supported(self, layer_type: str) -> bool:
        return NotImplementedError()

    def is_activation_supported(self, layer_type: str) -> bool:
        return layer_type.lower() == 'relu' or layer_type.lower() == 'activation' or layer_type.lower() == 'clamp'

    def is_quantization_fusing_supported(self, parent_layer, child_layer):
        if len(parent_layer.outputs) == 0 or parent_layer.outputs[0].layer.id != child_layer.id:
            # not supported fuse, let's ignore
            return False

        return self.is_quantization_supported(parent_layer.type) and \
            len(parent_layer.outputs) == 1 and \
            len(parent_layer.outputs[0].layer.inputs) == 1 and \
            self.is_activation_supported(child_layer.type)

    def get_quantization_layers(self) -> list:
        collect_layers = set()

        network_info = NetworkInfo(self._configuration.model)
        previous_previous_layer = None
        previous_layer = None
        layer_index = 0
        for layer in network_info.layers.values():
            if previous_previous_layer:
                if previous_layer and self.is_quantization_supported(previous_layer.type):
                    if self.is_quantization_fusing_supported(previous_layer, layer):
                        collect_layers.add(layer.name)
                    else:
                        collect_layers.add(previous_layer.name)
                    collect_layers.add(previous_previous_layer.name)

                if self.is_quantization_supported(layer.type) and layer_index == (len(network_info.layers) - 1):
                    collect_layers.add(layer.name)
                    collect_layers.add(previous_layer.name)

            layer_index += 1
            previous_previous_layer = previous_layer
            previous_layer = layer

        return collect_layers
