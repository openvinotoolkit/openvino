from collections import namedtuple
import multiprocessing
import threading

import openvino.inference_engine as ie

from ...utils.network_info import NetworkInfo
from ...network import Network

from ..layer_accuracy_drop_info import LayerAccuracyDropInfo
from ..logging import info, debug
from ..single_layer_network import SingleLayerNetwork
from ..inference_result import InferenceResult

QuantizationLayer = namedtuple('QuantizationLayer', 'index layer')


class SingleLayerNetworkThread(threading.Thread):
    def __init__(
        self,
        base_calibrator,
        statistics,
        full_network_result: InferenceResult,
        network: ie.IENetwork,
        network_info: NetworkInfo,
        quantization_layer: QuantizationLayer
    ):

        threading.Thread.__init__(self)
        self.base_calibrator = base_calibrator
        self.statistics = statistics
        self.full_network_result = full_network_result
        self.network = network
        self.network_info = network_info
        self.quantization_layer = quantization_layer
        self.result = None

    def run(self):
        self.result = self.base_calibrator.collect_in_thread(
            self.statistics,
            self.full_network_result,
            self.network,
            self.network_info,
            self.quantization_layer)

class CollectorByLayer:

    def __init__(self, configuration, plugin, normalizer):
        self._configuration = configuration
        self._plugin = plugin
        self._normalizer = normalizer

    def collect(self, statistics: dict(), full_network_result: InferenceResult) -> list:
        '''
        Method get layers which can be quantized and affect on final accuracy. Separate network is created for each layer.
        '''
        accuracy_drop_by_layer = list()

        network = ie.IENetwork(self._configuration.model, self._configuration.weights)
        # if self._configuration.batch_size:
        #     # need to use reshape API
        #     network.batch_size = self._configuration.batch_size

        try:
            network_info = NetworkInfo(self._configuration.model)

            #  2. go over all layers which affect accuracy and create network basing on it
            quantization_layers = list()

            index = 1
            threads = list()
            for layer in network.layers.values():
                if self._normalizer.is_quantization_supported(layer.type):
                    layer_info = network_info.get_layer(layer.name)
                    if (len(layer_info.outputs) == 1) and (len(layer_info.outputs[0].layer.inputs) == 1):
                        quantization_layer = QuantizationLayer(index, layer)
                        quantization_layers.append(quantization_layer)
                        threads.append(SingleLayerNetworkThread(self, statistics, full_network_result, network, network_info, quantization_layer))
                        index += 1

            it = iter(threads)
            threads_num = multiprocessing.cpu_count() * 2
            active_threads = list()
            while True:
                active_threads.clear()
                for thread_num in range(threads_num):
                    active_thread = next(it, None)
                    if not active_thread:
                        break
                    active_threads.append(active_thread)
                    active_thread.start()

                for active_thread in active_threads:
                    active_thread.join()

                if not active_thread:
                    debug("all layer networks were infered")
                    break

                debug("all layer networks before #{} were infered".format(active_thread.quantization_layer.index))

            for thread in threads:
                thread.join()
                accuracy_drop_by_layer.append(thread.result)

            accuracy_drop_by_layer.sort(key=lambda accuracy_drop: accuracy_drop.value, reverse=True)
            return accuracy_drop_by_layer
        finally:
            del network

    def collect_in_thread(
        self,
        statistics: dict(),
        full_network_result: InferenceResult,
        network: ie.IENetwork,
        network_info: NetworkInfo,
        quantization_layer: QuantizationLayer
    ) -> LayerAccuracyDropInfo:

        index = quantization_layer.index
        layer_to_clone = quantization_layer.layer
        layer_to_clone_info = network_info.get_layer(layer_to_clone.name)

        activation_layer = network.layers[layer_to_clone_info.outputs[0].layer.name] if (len(layer_to_clone_info.outputs) == 1 and self._normalizer.is_quantization_fusing_supported(layer_to_clone_info, layer_to_clone_info.outputs[0].layer)) else None
        if activation_layer:
            debug("create network #{} for layer {} ({}) -> {} ({})".format(index, layer_to_clone.name, layer_to_clone.type, activation_layer.name, activation_layer.type))
        else:
            debug("create network #{} for layer {} ({})".format(index, layer_to_clone.name, layer_to_clone.type))

        layer_network, reference_output_layer_name = self._normalizer.create_network_for_layer(
            self._configuration.weights,
            layer_to_clone,
            layer_to_clone_info,
            activation_layer)

        Network.reshape(layer_network, self._configuration.batch_size)

        network_stats = {}
        # TODO: initialize only neccessary statistic
        for layer_name, node_statistic in statistics.items():
            network_stats[layer_name] = ie.LayerStats(min=tuple(node_statistic.min_outputs), max=tuple(node_statistic.max_outputs))
        layer_network.stats.update(network_stats)

        params = layer_network.layers[layer_to_clone.name].params
        params["quantization_level"] = 'I8' if self._configuration.precision == 'INT8' else self._configuration.precision
        layer_network.layers[layer_to_clone.name].params = params

        exec_network = self._plugin.load(network=layer_network, config={ "EXCLUSIVE_ASYNC_REQUESTS": "YES" })

        if len(layer_network.inputs) != 1:
            raise ValueError("created network has several inputs")

        network_input_layer_name = next(iter(layer_network.inputs.keys()))

        with SingleLayerNetwork(
            network=layer_network,
            exec_network=exec_network,
            input_layer_name=network_input_layer_name,
            layer_name=layer_to_clone.name,
            output_layer_name=layer_to_clone.name + "_",
            reference_output_layer_name=reference_output_layer_name
        ) as single_layer_network:

            debug("single layer #{} {} network infer".format(index, single_layer_network.layer_name))
            accuracy_drop_list = self.infer_single_layer_network(single_layer_network, full_network_result)

            return LayerAccuracyDropInfo(
                layer_name=single_layer_network.layer_name,
                value=LayerAccuracyDropInfo.calculate(accuracy_drop_list))

    def infer_single_layer_network(self, single_layer_network: SingleLayerNetwork, full_network_results: list()):
        '''
        Native infer and compare results
        '''

        if full_network_results.result is None:
            raise ValueError("output inference results are absent")

        accuracy_drop_list = list()
        for full_network_result in full_network_results.result:
            difference = self._normalizer.infer_single_layer_network(single_layer_network, full_network_result)
            accuracy_drop_list.append(difference)

        return accuracy_drop_list
