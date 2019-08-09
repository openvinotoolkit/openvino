import openvino.inference_engine as ie

from ...utils.network_info import NetworkInfo
from ...network import Network

from ..layer_accuracy_drop_info import LayerAccuracyDropInfo
from ..logging import debug
from ..single_layer_network import SingleLayerNetwork
from ..inference_result import InferenceResult


class CollectorByImage:
    def __init__(self, configuration, plugin, normalizer, ignore_layer_names: list):
        self._configuration = configuration
        self._plugin = plugin
        self._normalizer = normalizer
        self._ignore_layer_names = ignore_layer_names

    def _create_single_layer_networks(self, stat):
        '''
        Method get layers which can be quantized and affect on final accuracy. Separate network is created for each layer.
        '''
        network = ie.IENetwork(self._configuration.model, self._configuration.weights)
        # if self._configuration.batch_size:
        #     # need to use reshape API
        #     network.batch_size = self._configuration.batch_size

        try:
            network_info = NetworkInfo(self._configuration.model)

            # CVS-14302: IE Network INT8 Normalizer: scale factor calculation is incorrect
            # for layer_name, layer_statistics in stat.items():
            #     layer_info = network_info.get_layer(layer_name)
            #     if layer_info.type == 'Convolution' and \
            #         layer_info.outputs and \
            #         layer_info.outputs[0].layer.type == 'ReLU' and \
            #         layer_info.outputs[0].layer.outputs[0] and \
            #         len(layer_statistics.max_outputs) > len(stat[layer_info.outputs[0].layer.name].max_outputs):

            #         relu_max_outputs = stat[layer_info.outputs[0].layer.name].max_outputs
            #         relu_min_outputs = stat[layer_info.outputs[0].layer.name].min_outputs

            #         while len(layer_statistics.max_outputs) > len(relu_max_outputs):
            #             relu_max_outputs.append(relu_max_outputs[-1])
            #             relu_min_outputs.append(relu_min_outputs[-1])

            single_layer_networks = dict()

            layer_index = 1
            for layer_to_clone in network.layers.values():
                layer_to_clone_info = network_info.get_layer(layer_to_clone.name)
                if layer_to_clone.name in self._ignore_layer_names or \
                        not self._normalizer.is_quantization_supported(layer_to_clone.type) or \
                        len(layer_to_clone_info.outputs) != 1 or \
                        len(layer_to_clone_info.outputs[0].layer.inputs != 1):
                    continue

                activation_layer = network.layers[layer_to_clone_info.outputs[0].layer.name] if (len(layer_to_clone_info.outputs) == 1 and self._normalizer.is_quantization_fusing_supported(layer_to_clone_info, layer_to_clone_info.outputs[0].layer)) else None
                if activation_layer:
                    debug("create network #{} for layer {} ({}) -> {} ({})".format(layer_index, layer_to_clone.name, layer_to_clone.type, activation_layer.name, activation_layer.type))
                else:
                    debug("create network #{} for layer {} ({})".format(layer_index, layer_to_clone.name, layer_to_clone.type))

                layer_network, reference_output_layer_name = self._normalizer.create_network_for_layer(
                    self._configuration.weights,
                    layer_to_clone,
                    layer_to_clone_info,
                    activation_layer)

                Network.reshape(layer_network, self._configuration.batch_size)

                network_stats = {}
                # TODO: initialize only neccessary statistic
                for layer_name, node_statistic in stat.items():
                    network_stats[layer_name] = ie.LayerStats(min=tuple(node_statistic.min_outputs), max=tuple(node_statistic.max_outputs))
                layer_network.stats.update(network_stats)

                params = layer_network.layers[layer_to_clone.name].params
                params["quantization_level"] = 'I8' if self._configuration.precision == 'INT8' else self._configuration.precision
                layer_network.layers[layer_to_clone.name].params = params

                exec_network = self._plugin.load(network=layer_network, config={ "EXCLUSIVE_ASYNC_REQUESTS": "YES" })

                if len(layer_network.inputs) != 1:
                    raise ValueError("created network has several inputs")

                network_input_layer_name = next(iter(layer_network.inputs.keys()))

                single_layer_networks[layer_to_clone.name] = SingleLayerNetwork(
                    network = layer_network,
                    exec_network = exec_network,
                    input_layer_name = network_input_layer_name,
                    layer_name = layer_to_clone.name,
                    output_layer_name = layer_to_clone.name + "_",
                    reference_output_layer_name = reference_output_layer_name)

                layer_index += 1

            return single_layer_networks
        finally:
            del network

    def collect(self, statistics: dict, full_network_results: InferenceResult) -> list:
        single_layer_networks = self._create_single_layer_networks(statistics)

        accuracy_drop_list_by_layer_name = dict()
        image_index = 1
        for full_network_result in full_network_results.result:
            debug("image {}/{} handling".format(image_index, full_network_results.result.size()))

            for single_layer_network_name, single_layer_network in single_layer_networks.items():
                accuracy_drop = self._normalizer.infer_single_layer_network(single_layer_network, full_network_result)

                if single_layer_network_name not in accuracy_drop_list_by_layer_name:
                    accuracy_drop_list_by_layer_name[single_layer_network_name] = list()

                accuracy_drop_list_by_layer_name[single_layer_network_name].append(accuracy_drop)
            image_index += 1

        accuracy_drop_by_layer = list()
        for layer_name, accuracy_drop_list in accuracy_drop_list_by_layer_name.items():
            accuracy_drop_by_layer.append(LayerAccuracyDropInfo(
                layer_name=layer_name,
                value=LayerAccuracyDropInfo.calculate(accuracy_drop_list)))

        single_layer_network.release()
        single_layer_networks.clear()

        accuracy_drop_by_layer.sort(key=lambda accuracy_drop: accuracy_drop.value, reverse=True)
        return accuracy_drop_by_layer
