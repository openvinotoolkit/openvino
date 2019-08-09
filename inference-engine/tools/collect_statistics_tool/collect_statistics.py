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
from openvino.tools.calibration import CalibratorConfiguration, CalibrationConfigurationHelper, CalibratorFactory, CommandLineProcessor
from openvino.tools.utils import Path

def collect_statistics():
    with CommandLineProcessor.process() as configuration:
        calibrator = CalibratorFactory.create(configuration.precision, CalibratorConfiguration(configuration))

        print("Collecting FP32 statistics for {}...".format(configuration.model))
        fp32_result = calibrator.infer(add_outputs=True, collect_aggregated_statistics=True)
        print("FP32 accuracy: {0:.4f}{1}".format(fp32_result.metrics.accuracy.value, fp32_result.metrics.accuracy.symbol))

        output_model_file_path = Path.get_model(configuration.output_model, "_statistics")
        output_weights_file_path = Path.get_weights(configuration.output_weights, "_statistics")

        quantization_levels = calibrator.get_quantization_levels(CalibrationConfigurationHelper.read_ignore_layer_names(configuration))
        statistics = fp32_result.aggregated_statistics.get_node_statistics()
        calibrator.save(output_model_file_path, output_weights_file_path, quantization_levels, statistics)
        print("Network with statistics was written to {}.(xml|bin) IR file".format(os.path.splitext(output_model_file_path)[0]))

if __name__ == '__main__':
    collect_statistics()
