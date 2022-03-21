# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import math

import cv2
import numpy as np

from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, save_model, compress_model_weights, create_pipeline
from openvino.tools.pot.utils.logger import init_logger
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser

# Initialize the logger to print the quantization process in the console.
init_logger(level='INFO')


_SEGMENTATION_COLORS = ((
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
))


# Custom DataLoader class implementation that is required for
# the proper reading of Pascal VOC Segmentation images and annotations.
class VOCSegmentationLoader(DataLoader):

    # Required methods:
    def __init__(self, config):
        super().__init__(config)
        self._image_size = self.config.image_size
        self._img_ids = self._read_img_ids(self.config)

    def __getitem__(self, index):
        """
        Returns annotation and image (and optionally image metadata) at the specified index.
        Possible formats:
        (img_id, img_annotation), image
        (img_id, img_annotation), image, image_metadata
        """

        if index >= len(self):
            raise IndexError
        mask_path = os.path.join(self.config.mask_dir, self._img_ids[index] + '.png')
        image_path = os.path.join(self.config.data_source, self._img_ids[index] + '.jpg')

        return self._read_and_preprocess_image(image_path), self._read_and_preprocess_mask(mask_path)

    def __len__(self):
        """ Returns size of the dataset """
        return len(self._img_ids)

    # Methods specific to the current implementation
    @staticmethod
    def _read_img_ids(config):
        with open(config.imageset_file) as f:
            return f.read().splitlines()

    def _read_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)

        # OpenCV returns image in BGR format. Convert  BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pad image to destination size
        image = central_padding(image, self._image_size, self._image_size)

        return image

    def _read_and_preprocess_mask(self, mask_path):
        mask = self._read_and_preprocess_image(mask_path)
        encoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for label, color in enumerate(_SEGMENTATION_COLORS):
            encoded_mask[np.where(np.all(mask == color, axis=-1))[:2]] = label

        return encoded_mask


# Custom implementation of Mean Intersection Over Union metric.
class MeanIOU(Metric):

    # Required methods
    def __init__(self, num_classes):
        self._classes_num = num_classes
        super().__init__()
        self._name = 'mean_iou'
        self._current_cm = []
        self._total_cm = np.zeros((self._classes_num, self._classes_num))

    @property
    def avg_value(self):
        """ Returns average metric value for all model outputs.
        Possible format: {metric_name: metric_value}
        """
        return {self._name: self._evaluate(self._total_cm)}

    # pylint: disable=E1101
    def update(self, output, target):
        """ Calculates and updates metric value
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The Mean IOU metric cannot be calculated '
                            'for a model with multiple outputs')
        self._current_cm = []
        y_pred = output[0].flatten()
        y_true = target[0].flatten()
        valid_pixels = (y_true >= 0) & (y_true < self._classes_num) & \
                       (y_pred >= 0) & (y_pred < self._classes_num)
        y_true = y_true[valid_pixels]
        y_pred = y_pred[valid_pixels]
        current_cm = np.bincount(self._classes_num * y_true + y_pred,
                                 minlength=self._classes_num ** 2)
        current_cm = current_cm.reshape(self._classes_num, self._classes_num)
        self._current_cm.append(current_cm)
        self._total_cm += current_cm

    def reset(self):
        """ Resets metric """
        self._current_cm = []
        self._total_cm = np.zeros((self._classes_num, self._classes_num))

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'mean_iou'}}

    # Methods specific to the current implementation
    @staticmethod
    def _evaluate(confusion_matrix):
        intersection = np.diagonal(confusion_matrix)
        union = confusion_matrix.sum(axis=1) + \
                confusion_matrix.sum(axis=0) - \
                intersection

        return np.nanmean(np.divide(intersection, union,
                                    out=np.full(intersection.shape, np.nan),
                                    where=union != 0))


def central_padding(image, dst_height, dst_width, value=(0, 0, 0), channels_first=False):
    h, w = image.shape[-2:] if channels_first else image.shape[:2]
    pad = [max(0, math.floor((dst_height - h) / 2.0)),
           max(0, math.floor((dst_width - w) / 2.0))]
    pad.extend([max(0, dst_height - h - pad[0]),
                max(0, dst_width - w - pad[1])])

    return cv2.copyMakeBorder(image, pad[0], pad[2], pad[1], pad[3],
                              cv2.BORDER_CONSTANT, value=value)


def main():
    parser = get_common_argparser()
    parser.add_argument(
        '--mask-dir',
        help='Path to the directory with segmentation masks',
        required=True
    )
    parser.add_argument(
        '--imageset-file',
        help='Path to the ImageSet file',
        required=True
    )

    args = parser.parse_args()
    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    model_config = {
        'model_name': 'deeplabv3',
        'model': os.path.expanduser(args.model),
        'weights': os.path.expanduser(args.weights)
    }

    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 4,
        'eval_requests_number': 4
    }

    dataset_config = {
        'data_source': os.path.expanduser(args.dataset),
        'mask_dir': os.path.expanduser(args.mask_dir),
        'imageset_file': os.path.expanduser(args.imageset_file),
        'image_size': 513
    }

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': 'ANY',
                'preset': 'performance',
                'stat_subset_size': 300
            }
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    data_loader = VOCSegmentationLoader(dataset_config)

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = MeanIOU(num_classes=21)

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config,
                      data_loader=data_loader,
                      metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6: Execute the pipeline.
    compressed_model = pipeline.run(model)

    # Step 7 (Optional): Compress model weights to quantized precision
    #                    in order to reduce the size of final .bin file.
    compress_model_weights(compressed_model)

    # Step 8: Save the compressed model to the desired path.
    save_model(compressed_model, os.path.join(os.path.curdir, 'optimized'))

    # Step 9 (Optional): Evaluate the compressed model. Print the results.
    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        for name, value in metric_results.items():
            print('{: <27s}: {}'.format(name, value))


if __name__ == '__main__':
    main()
