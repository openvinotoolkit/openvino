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

import json
import os
import platform

from ..utils.network_info import NetworkInfo

from ..benchmark.benchmark import Benchmark
from ..network import Network

from .logging import info, debug, info_performance_counters, info_layer_accuracy_drop
from .calibrator_configuration import CalibratorConfiguration
from .calibrator_factory import CalibratorFactory
from .calibration_configuration import CalibrationConfiguration, CalibrationConfigurationHelper
from .layer_accuracy_drop.collector_by_layer import CollectorByLayer

class Calibrator:
    def __init__(self, configuration: CalibrationConfiguration):
        if configuration is None:
            raise ValueError("configuration is None")

        self._configuration = configuration

    def run(self) -> Network:
        calibrator = CalibratorFactory.create(
            self._configuration.precision,
            CalibratorConfiguration(self._configuration))
        benchmark = Benchmark(self._configuration)

        info("Processor: {}".format(platform.processor()))

        info("Collecting FP32 statistics for {}...".format(self._configuration.model))
        fp32_result = calibrator.infer(
            add_outputs=True,
            collect_aggregated_statistics=True,
            collect_performance_counters=True)
        fp32_accuracy = fp32_result.metrics.accuracy
        fp32_latency = benchmark.run(iterations_count=self._configuration.benchmark_iterations_count).latency
        info("FP32 accuracy: {0:.4f}%, latency: {1:0.4f} ms".format(100.0 * fp32_accuracy, 1000 * fp32_latency))

        info("FP32 performance counters:\n")
        info_performance_counters(fp32_result.performance_counters)

        ignore_layer_names = CalibrationConfigurationHelper.read_ignore_layer_names(self._configuration)
        fp32_result.aggregated_statistics.pop(ignore_layer_names=ignore_layer_names)
        fp32_aggregated_statistics = fp32_result.aggregated_statistics
        fp32_result = None

        info("Verification of network accuracy if all possible layers converted to {}\n".format(
		    self._configuration.precision))

        best_lp_accuracy = None
        best_lp_latency = 0.0
        best_lp_threshold = 100.0
        best_lp_statistics = None
        best_lp_performance_counters = None

        threshold = 100.0
        threshold_low_boundary = 95.0
        threshold_step = .5

        quantization_levels = calibrator.get_quantization_levels(ignore_layer_names)

        min_accuracy_drop = None
        while threshold >= threshold_low_boundary:
            info("Validate {} accuracy, threshold for activation statistics: {}%".format(
                self._configuration.precision,
                threshold))

            lp_statistics = fp32_aggregated_statistics.get_node_statistics(threshold)
            with Network.reload(
                model_path=self._configuration.model,
                statistics=lp_statistics,
                quantization_levels=quantization_levels,
                batch_size=self._configuration.batch_size
            ) as reloaded_network:

                with calibrator.infer(network=reloaded_network.ie_network,
                                      collect_performance_counters=True) as lp_result:
                    lp_accuracy = lp_result.metrics.accuracy
                    lp_performance_counters = lp_result.performance_counters
                    lp_latency = benchmark.run(
                        network=reloaded_network,
                        iterations_count=self._configuration.benchmark_iterations_count).latency

            if best_lp_accuracy is None or lp_accuracy > best_lp_accuracy:

                best_lp_accuracy = lp_accuracy
                best_lp_latency = lp_latency
                best_lp_threshold = threshold
                if best_lp_statistics:
                    del best_lp_statistics
                best_lp_statistics = lp_statistics
                best_lp_performance_counters = lp_performance_counters
            else:
                del lp_statistics

            min_accuracy_drop = fp32_accuracy - lp_accuracy if min_accuracy_drop is None else min(
                min_accuracy_drop,
                fp32_accuracy - lp_accuracy)

            info("{0} accuracy is {1:.4f}%, latency: {2:0.4f} ms\n".format(
                self._configuration.precision,
                100.0 * lp_accuracy,
                1000.0 * lp_latency))
            threshold = threshold - threshold_step


        info("Best {0} accuracy is {1:.4f}%, latency: {2:0.4f} ms for threshold {3}%".format(
            self._configuration.precision,
            100.0 * best_lp_accuracy,
            1000.0 * best_lp_latency,
            best_lp_threshold))

        info("{} performance counters:\n".format(self._configuration.precision))
        info_performance_counters(best_lp_performance_counters)

        accuracy_was_satisfied = False
        if (fp32_accuracy - best_lp_accuracy) > (self._configuration.threshold / 100):
            info("Accuracy of all layers conversion does not correspond to the required threshold")
            info(("FP32 Accuracy: {0:.4f}% (latency: {1:0.4f} ms) vs all low precision layers accuracy: {2:.4f}% "
                  "(latency: {3:0.4f} ms), threshold for activation statistics: {4}%").format(100.0 * fp32_accuracy,
                                                                                              1000.0 * fp32_latency,
                                                                                              100.0 * best_lp_accuracy,
                                                                                              1000.0 * best_lp_latency,
                                                                                              best_lp_threshold))

            info("Collecting all raw FP32 results")

            quantization_layers = calibrator.get_quantization_layers()
            debug("{} layers (total {}) are selected to cache".format(
                len(quantization_layers),
                len(NetworkInfo(self._configuration.model).layers)))

            with calibrator.infer(add_outputs=True,
                                  collect_resuls=True,
                                  collect_layers=quantization_layers) as fp32_result_with_raw_data:
                info("Collecting intermediate per-layer accuracy drop")
                layers_accuracy_drop = CollectorByLayer(
                    self._configuration,
                    calibrator.plugin,
                    calibrator).collect(best_lp_statistics, fp32_result_with_raw_data)

                info("Layer accuracy drop:\n")
                info_layer_accuracy_drop(layers_accuracy_drop)

            if layers_accuracy_drop:
                info("Starting to reduce number of layers being converted to Int8")

                for layer_accuracy_drop in layers_accuracy_drop:
                    info("Returning of '{}' to FP32 precision, start validation".format(layer_accuracy_drop.layer_name))
                    quantization_levels[layer_accuracy_drop.layer_name] = "FP32"

                    with Network.reload(
                        self._configuration.model,
                        statistics=best_lp_statistics,
                        quantization_levels=quantization_levels,
                        batch_size=self._configuration.batch_size
                    ) as reloaded_network:

                        with calibrator.infer(network=reloaded_network.ie_network) as layer_int8_result:
                            best_lp_accuracy = layer_int8_result.metrics.accuracy
                            best_lp_latency = benchmark.run(
                                network=reloaded_network,
                                iterations_count=self._configuration.benchmark_iterations_count).latency

                    accuracy_drop = fp32_accuracy - best_lp_accuracy
                    min_accuracy_drop = accuracy_drop if min_accuracy_drop is None else min(min_accuracy_drop,
                                                                                            accuracy_drop)
                    if accuracy_drop > (self._configuration.threshold / 100.0):
                        info("Was not achieved: FP32 accuracy: {0:.4f}% (latency: {1:.4} ms) VS {2} accuracy: {3:.4f}% "
                             "(latency {4:.4f} ms), accuracy drop {5:.4f}%".format(100.0 * fp32_accuracy,
                                                                                   1000.0 * fp32_latency,
                                                                                   self._configuration.precision,
                                                                                   100.0 * best_lp_accuracy,
                                                                                   1000.0 * best_lp_latency,
                                                                                   100.0 * accuracy_drop))
                    else:
                        accuracy_was_satisfied = True
                        info("Achieved: FP32 accuracy: {0:.4f}% (latency: {1:.4} ms) VS {2} accuracy: {3:.4}% "
                             "(latency: {4:.4} ms), accuracy drop {5:.4}%".format(100.0 * fp32_accuracy,
                                                                                  1000.0 * fp32_latency,
                                                                                  self._configuration.precision,
                                                                                  100.0 * best_lp_accuracy,
                                                                                  1000.0 * best_lp_latency,
                                                                                  100.0 * accuracy_drop))
                        break
            else:
                info("No layers to reduce number of converted to Int8")

        else:
            accuracy_was_satisfied = True

        if accuracy_was_satisfied:
            info("Achieved required accuracy drop satisfying threshold")
            info("FP32 accuracy: {0:.4f}% (latency: {1:.4} ms) vs current low precision configuration accuracy: "
                 "{2:.4f}% (latency: {3:.4} ms) with threshold for activation statistic: {4}%".format(
                     100.0 * fp32_accuracy,
                     1000.0 * fp32_latency,
                     100.0 * best_lp_accuracy,
                     1000.0 * best_lp_latency,
                     best_lp_threshold))

            quantized_layers_count = 0
            for quantization_level in quantization_levels.values():
                if quantization_level != "FP32":
                    quantized_layers_count += 1
            info("quantized layers (quantized {}, total {} layers):".format(
                quantized_layers_count,
                len(quantization_levels)))

            layers_message = "FP32 layers:\n"
            for layer_name, quantization_level in quantization_levels.items():
                if quantization_level == "FP32":
                    layers_message += "\tlayer '{}': {}\n".format(layer_name, quantization_level)
            info(layers_message)

            layers_message = "{} layers:\n".format(self._configuration.precision)
            for layer_name, quantization_level in quantization_levels.items():
                if quantization_level != "FP32":
                    layers_message += "\tlayer '{}': {}\n".format(layer_name, quantization_level)
            info(layers_message)

            info("Write calibrated network to {}.(xml|bin) IR file".format(
                os.path.splitext(self._configuration.output_model)[0]))

            calibrator.save(
                self._configuration.output_model,
                self._configuration.output_weights,
                quantization_levels,
                best_lp_statistics)

            # TODO: need to load from hard drive while not fixed
            output_network = Network(self._configuration.output_model, self._configuration.output_weights)
            return output_network
        else:
            info("Required threshold of accuracy drop cannot be achieved with any {0} quantization. Minimal accuracy "
                 "drop: {1:0.4%}".format(self._configuration.precision, min_accuracy_drop))

            return None
