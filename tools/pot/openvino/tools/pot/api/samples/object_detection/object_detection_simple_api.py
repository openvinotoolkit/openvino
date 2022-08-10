# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from openvino.tools.pot import AccurracyAwareQuantizationParameters, quantize_with_accuracy_control, \
                                export
from openvino.tools.pot.utils.logger import init_logger
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser
from openvino.tools.pot.api.samples.object_detection.metric import MAP
from openvino.tools.pot.api.samples.object_detection.data_loader import COCOLoader

init_logger(level='INFO')


def main():
    parser = get_common_argparser()
    parser.add_argument(
        '--annotation-path',
        help='Path to the directory with annotation file',
        required=True
    )
    args = parser.parse_args()
    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    dataset_config = {
        'images_path': os.path.expanduser(args.dataset),
        'annotation_path': os.path.expanduser(args.annotation_path),
    }

    # Step 1: Initialize the data loader.
    data_loader = COCOLoader(dataset_config)
    # Step 2: Initialize the metric.
    metric = MAP(91, data_loader.labels)

    parameters = AccurracyAwareQuantizationParameters(model_name = 'ssd_mobilenet_v1_fpn',
                    model_path = args.model, weights_path = args.weights,
                    preset = 'mixed', max_drop = 0.004)

    # Step 3: Quantize model
    compressed_model = quantize_with_accuracy_control(data_loader, metric, parameters)

    # Step 4: Save quantized model
    export(compressed_model, os.path.join(os.path.curdir, 'optimized'))


if __name__ == '__main__':
    main()
