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
import os
import platform

from ..utils.network_info import NetworkInfo

from ..network import Network

from .benchmark_facade import BenchmarkFacade
from .logging import info, debug, info_performance_counters, info_layer_accuracy_drop
from .calibrator_configuration import CalibratorConfiguration
from .calibrator_factory import CalibratorFactory
from .calibration_configuration import CalibrationConfiguration, CalibrationConfigurationHelper
from .layer_accuracy_drop.collector_by_layer import CollectorByLayer
from .calibrator_result_types import *

import openvino.tools.statistics_collector as SC

class Calibrator:
    def __init__(self, configuration: CalibrationConfiguration):
        if configuration is None:
            raise ValueError("configuration is None")

        self._configuration = configuration
        if not self._configuration.simplified_mode:
            self._calibrator = CalibratorFactory.create(self._configuration.precision,
                                                        CalibratorConfiguration(configuration))
            self._benchmark = BenchmarkFacade(self._configuration.device, self._configuration.batch_size,
                                              self._configuration.benchmark_iterations_count, self._configuration.cpu_extension)
            self._ignore_layer_names = CalibrationConfigurationHelper.read_ignore_layer_names(self._configuration)
            self._quantization_levels = self._calibrator.get_quantization_levels(self._ignore_layer_names)

    def collect_fp32_results(self) -> RawResults:
        info("Processor: {}".format(platform.processor()))
        info("Collecting original network statistics for {}...".format(self._configuration.model))
        fp32_stats = self._calibrator.infer(
            add_outputs=True,
            collect_aggregated_statistics=True,
            collect_performance_counters=True)
        iterations = self._configuration.benchmark_iterations_count
        fp32_latency = 0.0
        if iterations > 0:
            fp32_latency = self._benchmark.run(self._configuration.model).latency
        accuracy = fp32_stats.metrics.accuracy
        info("Original network accuracy: {0:.4f}{1}, latency: {2:0.4f} ms".format(accuracy.value,
                                                                      accuracy.symbol,
                                                                      fp32_latency))
        info("Original network performance counters:\n")
        info_performance_counters(fp32_stats.performance_counters)
        return RawResults(fp32_stats=fp32_stats, fp32_latency=fp32_latency)

    def get_statistics(self, raw_results: RawResults) -> LowPrecisionResults:

        info("Verification of network accuracy if all possible layers converted to {}\n"
             .format(self._configuration.precision))
        fp32_result = raw_results.fp32_stats

        fp32_result.aggregated_statistics.pop(ignore_layer_names=self._ignore_layer_names)
        fp32_aggregated_statistics = fp32_result.aggregated_statistics

        fp32_accuracy = fp32_result.metrics.accuracy

        best_lp_stats = LowPrecisionResults()

        threshold = 100.0
        threshold_low_boundary = self._configuration.threshold_boundary
        threshold_step = self._configuration.threshold_step

        while threshold >= threshold_low_boundary:
            info("Validate {} accuracy, threshold for activation statistics: {}%".format(
                self._configuration.precision,
                threshold))
            lp_latency = best_lp_stats.latency

            lp_statistics = fp32_aggregated_statistics.get_node_statistics(threshold)
            tmp_model_path = Network.serialize_tmp_model(
                model_path=self._configuration.model,
                statistics=lp_statistics,
                quantization_levels=self._quantization_levels)

            with self._calibrator.infer(model_path=tmp_model_path,
                                        collect_performance_counters=True) as lp_result:
                lp_accuracy = lp_result.metrics.accuracy
                lp_performance_counters = lp_result.performance_counters
                iterations = self._configuration.benchmark_iterations_count
                if iterations > 0:
                    lp_latency = self._benchmark.run(tmp_model_path).latency
            Network.rm_tmp_location(tmp_model_path)

            if lp_accuracy.is_better(best_lp_stats.accuracy, fp32_accuracy):

                best_lp_stats.accuracy = lp_accuracy
                best_lp_stats.latency = lp_latency
                best_lp_stats.threshold = threshold

                if best_lp_stats.statistics:
                    del best_lp_stats.statistics
                best_lp_stats.statistics = lp_statistics
                best_lp_stats.performance_counters = lp_performance_counters

                best_lp_stats.accuracy_drop = lp_accuracy.calculate_drop(fp32_accuracy)
            else:
                del lp_statistics

            info("{0} accuracy is {1:.4f}{2}, latency: {3:0.4f} ms\n".format(
                self._configuration.precision,
                lp_accuracy.value,
                lp_accuracy.symbol,
                lp_latency))
            threshold = threshold - threshold_step

        info("Best {0} accuracy is {1:.4f}{2}, latency: {3:0.4f} ms for threshold {4}%".format(
            self._configuration.precision,
            best_lp_stats.accuracy.value,
            best_lp_stats.accuracy.symbol,
            best_lp_stats.latency,
            best_lp_stats.threshold))

        info("{} performance counters:\n".format(self._configuration.precision))
        info_performance_counters(best_lp_stats.performance_counters)

        best_lp_stats.accuracy_fits_threshold = best_lp_stats.accuracy.is_achieved(
            fp32_accuracy,
            self._configuration.threshold
        )

        return best_lp_stats

    def return_back_to_fp32(self, lp_results: LowPrecisionResults, raw_results: RawResults):
        info("Collecting all raw original precision results")

        quantization_layers = self._calibrator.get_quantization_layers()
        debug("{} layers (total {}) are selected to cache".format(
            len(quantization_layers),
            len(NetworkInfo(self._configuration.model).layers)))

        # collect raw original precision outputs per image and use each output
        # to calculate layer accuracy drop

        with self._calibrator.infer(
                add_outputs=True,
                collect_resuls=True,
                collect_layers=quantization_layers,
                per_layer_statistics=lp_results.statistics,
                ignore_layer_names=self._ignore_layer_names) as fp32_result_with_raw_data:
            if fp32_result_with_raw_data.layers_accuracy_drop:
                layers_accuracy_drop = fp32_result_with_raw_data.layers_accuracy_drop
            else:
                info("Collecting intermediate per-layer accuracy drop")
                layers_accuracy_drop = CollectorByLayer(
                    self._configuration,
                    self._calibrator.plugin,
                    self._calibrator,
                    self._ignore_layer_names)\
                    .collect(lp_results.statistics, fp32_result_with_raw_data)

            info("Layer accuracy drop:\n")
            info_layer_accuracy_drop(layers_accuracy_drop)

        if layers_accuracy_drop:
            info("Starting to reduce number of layers being converted to Int8")

            for layer_accuracy_drop in layers_accuracy_drop:
                info("Returning of '{}' to original precision, start validation".format(layer_accuracy_drop.layer_name))
                self._quantization_levels[layer_accuracy_drop.layer_name] = layer_accuracy_drop.precision
                best_lp_latency = 0.0

                tmp_model_path = Network.serialize_tmp_model(
                    model_path=self._configuration.model,
                    statistics=lp_results.statistics,
                    quantization_levels=self._quantization_levels)

                with self._calibrator.infer(model_path=tmp_model_path) as layer_int8_result:
                    lp_results.accuracy = layer_int8_result.metrics.accuracy
                    fp32_accuracy = raw_results.fp32_stats.metrics.accuracy
                    accuracy_drop = lp_results.accuracy.calculate_drop(fp32_accuracy)
                    iterations = self._configuration.benchmark_iterations_count
                    if iterations > 0:
                        best_lp_latency = self._benchmark.run(tmp_model_path).latency
                Network.rm_tmp_location(tmp_model_path)

                lp_results.accuracy_drop = accuracy_drop if accuracy_drop < lp_results.accuracy_drop else lp_results.accuracy_drop
                if not lp_results.accuracy.is_achieved(fp32_accuracy, self._configuration.threshold):
                    info("Was not achieved: original network accuracy: {0:.4f}{1} (latency: {2:.4} ms) VS {3} accuracy: {4:.4f}{5} "
                         "(latency {6:.4f} ms), accuracy drop {7:.4f}%"
                         .format(fp32_accuracy.value,
                                 fp32_accuracy.symbol,
                                 raw_results.fp32_latency,
                                 self._configuration.precision,
                                 lp_results.accuracy.value,
                                 lp_results.accuracy.symbol,
                                 best_lp_latency,
                                 accuracy_drop))

                else:
                    lp_results.accuracy_fits_threshold = True
                    info("Achieved: original network accuracy: {0:.4f}{1} (latency: {2:.4} ms) VS {3} accuracy: {4:.4}{5} "
                         "(latency: {6:.4} ms), accuracy drop {7:.4}%"
                         .format(fp32_accuracy.value,
                                 fp32_accuracy.symbol,
                                 raw_results.fp32_latency,
                                 self._configuration.precision,
                                 lp_results.accuracy.value,
                                 lp_results.accuracy.symbol,
                                 best_lp_latency,
                                 lp_results.accuracy_drop))

                    break
        else:
            info("No layers to reduce number of converted to Int8")

    def save(self, best_lp_statistics):

        self._calibrator.save(
            self._configuration.output_model,
            self._configuration.output_weights,
            self._quantization_levels,
            best_lp_statistics)

        # TODO: need to load from hard drive while not fixed
        output_network = Network(self._configuration.output_model, self._configuration.output_weights)
        return output_network

    def run(self) -> Network:
        if self._configuration.simplified_mode:
            sc = SC.StatisticsCollector(deviceName = self._configuration.device[0],
                                        custom_cpu_library = self._configuration.cpu_extension,
                                        custom_cldnn = '',
                                        modelFilePath = self._configuration.model,
                                        imagesPath = str(self._configuration.config.source),
                                        img_number = self._configuration.config.subset,
                                        batch = self._configuration._batch_size,
                                        progress = self._configuration.progress)
            sc.collectStatisticsToIR(self._configuration._output_model, self._configuration.precision)
            return None

        raw_results = self.collect_fp32_results()

        lp_results = self.get_statistics(raw_results)

        if not lp_results.accuracy_fits_threshold:
            info("Accuracy of all layers conversion does not correspond to the required threshold")
            info(("Original network accuracy: {0:.4f}{1} (latency: {2:0.4f} ms) vs all low precision layers accuracy: {3:.4f}{4} "
                  "(latency: {5:0.4f} ms), threshold for activation statistics: {6}%")
                 .format(raw_results.fp32_stats.metrics.accuracy.value,
                         raw_results.fp32_stats.metrics.accuracy.symbol,
                         raw_results.fp32_latency,
                         lp_results.accuracy.value,
                         lp_results.accuracy.symbol,
                         lp_results.latency,
                         lp_results.threshold))
            self.return_back_to_fp32(lp_results, raw_results)

        if lp_results.accuracy_fits_threshold:
            info("Achieved required accuracy drop satisfying threshold")
            info("Original network accuracy: {0:.4f}{1} (latency: {2:.4} ms) vs current low precision configuration accuracy: "
                 "{3:.4f}{4} (latency: {5:.4} ms) with threshold for activation statistic: {6}%".format(
                    raw_results.fp32_stats.metrics.accuracy.value,
                    raw_results.fp32_stats.metrics.accuracy.symbol,
                    raw_results.fp32_latency,
                    lp_results.accuracy.value,
                    lp_results.accuracy.symbol,
                    lp_results.latency,
                    lp_results.threshold))

            quantized_layers_count = 0
            for quantization_level in self._quantization_levels.values():
                if quantization_level == self.get_cut_precision_name():
                    quantized_layers_count += 1
            info("quantized layers (quantized {}, total {} layers):".format(
                quantized_layers_count,
                len(self._quantization_levels)))

            layers_message = "Original precision layers:\n"
            for layer_name, quantization_level in self._quantization_levels.items():
                if quantization_level != self.get_cut_precision_name():
                    layers_message += "\tlayer '{}': {}\n".format(layer_name, quantization_level)
            info(layers_message)

            layers_message = "{} layers:\n".format(self._configuration.precision)
            for layer_name, quantization_level in self._quantization_levels.items():
                if quantization_level == self.get_cut_precision_name():
                    layers_message += "\tlayer '{}': {}\n".format(layer_name, quantization_level)
            info(layers_message)

            info("Write calibrated network to {}.(xml|bin) IR file".format(
                os.path.splitext(self._configuration.output_model)[0]))

            return self.save(lp_results.statistics)
        else:
            info("Required threshold of accuracy drop cannot be achieved with any {0} quantization. Minimal accuracy "
                 "drop: {1:0.4}%".format(self._configuration.precision, lp_results.accuracy_drop))
            return None

    def get_cut_precision_name(self):
        full_name = self._configuration.precision
        return full_name[0] + full_name[-1]

