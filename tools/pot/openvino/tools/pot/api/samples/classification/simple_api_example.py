# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from inspect import Parameter
import os
from sqlite3.dbapi2 import _Parameters
import numpy as np
from cv2 import imread, resize as cv2_resize

from openvino.tools.pot import quantize_post_training, QuantizationParameters, export
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser

from .classification_sample import ImageNetDataLoader

def main():
    argparser = get_common_argparser()
    argparser.add_argument(
        '-a',
        '--annotation-file',
        help='File with Imagenet annotations in .txt format',
        required=True
    )

    args = argparser.parse_args()

    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    dataset_config = {
        'data_source': os.path.expanduser(args.dataset),
        'annotation_file': os.path.expanduser(args.annotation_file),
        'has_background': True,
        'preprocessing': [
            {
                'type': 'crop',
                'central_fraction': 0.875
            },
            {
                'type': 'resize',
                'width': 224,
                'height': 224
            }
        ],
    }

    # Create a user-defined DataLoader object
    data_loader = ImageNetDataLoader(dataset_config)

    # Define mandatory quantization parameters
    parameters = QuantizationParameters()
    parameters.model_name = 'sample_model'
    parameters.model_path = os.path.expanduser(args.model),
    parameters.weights_path = os.path.expanduser(args.weights)

    # Quantize model
    compressed_model = quantize_post_training(parameters, data_loader)

    # Save quantized model
    export(compressed_model, os.path.join(os.path.curdir, 'optimized'))

if __name__ == '__main__':
    main()
