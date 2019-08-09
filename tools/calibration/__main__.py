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

from argparse import ArgumentParser

import openvino.tools.calibration as calibration
import openvino.tools.utils as utils


def calibrate():
    config = calibration.CommandLineReader.read()
    network = calibration.Calibrator(config).run()
    network.serialize(config.output_model)


def check_accuracy():
    config = calibration.CommandLineReader.read()
    calibrator = calibration.CalibratorFactory.create(config.precision, calibration.CalibratorConfiguration(config))

    print("Collecting accuracy for {}...".format(config.model))
    result = calibrator.infer()
    print("Accuracy: {0:.4f}%".format(100.0 * result.metrics.accuracy))


def collect_statistics():
    import os
    config = calibration.CommandLineReader.read()
    calibrator = calibration.CalibratorFactory.create(config.precision, calibration.CalibratorConfiguration(config))

    print("Collecting original network statistics for {}...".format(config.model))
    fp32_result = calibrator.infer(add_outputs=True, collect_aggregated_statistics=True)
    print("Original network accuracy: {0:.4f}%".format(100.0 * fp32_result.metrics.accuracy))

    output_model_file_path = \
        os.path.splitext(config.model)[0] + ("_{}_statistics_without_ignored.xml".format(config.precision.lower()) if
                                             config.ignore_layer_names else
                                             "_{}_statistics.xml".format(config.precision.lower()))
    output_weights_file_path = utils.Path.get_weights(output_model_file_path)

    quantization_levels = \
        calibrator.get_quantization_levels(calibration.CalibrationConfigurationHelper.read_ignore_layer_names(config))
    statistics = fp32_result.aggregated_statistics.get_node_statistics()
    calibrator.save(output_model_file_path, output_weights_file_path, quantization_levels, statistics)
    print("Network with statistics was written to {}.(xml|bin) IR file".format(os.path.splitext(output_model_file_path)[0]))


def __build_arguments_parser():
    parser = ArgumentParser(description='Calibration Tool')
    parser.add_argument(
        'action',
        help='Optional, possible values: calibrate, collect_statistics or check_accuracy',
        nargs='?',
        choices=('calibrate', 'collect_statistics', 'check_accuracy'))
    return parser


if __name__ == '__main__':
    parser, unknown_args = __build_arguments_parser().parse_known_args()
    if parser.action == 'calibrate':
        calibrate()
    elif parser.action == 'collect_statistics':
        collect_statistics()
    elif parser.action == 'check_accuracy':
        check_accuracy()
    else:
        calibrate()
