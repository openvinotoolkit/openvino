# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import nibabel as nib
import numpy as np
from scipy.ndimage import interpolation

from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, save_model, compress_model_weights, create_pipeline
from openvino.tools.pot.utils.logger import init_logger
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser

# Initialize the logger to print the quantization process in the console.
init_logger(level='INFO')


# Custom DataLoader class implementation that is required for
# the proper reading of BRATS 3D Segmentation images and annotations.
class BRATSDataLoader(DataLoader):

    # Required methods:
    def __init__(self, config):
        super().__init__(config)
        self._img_ids = sorted(os.listdir(self.config.data_source))

    def __getitem__(self, index):
        """
        Returns annotation, image and image metadata at the specified index.
        Possible formats:
        (img_id, img_annotation), image
        (img_id, img_annotation), image, image_metadata
        """

        if index >= len(self):
            raise IndexError
        mask_path = os.path.join(self.config.mask_dir, self._img_ids[index])
        image_path = os.path.join(self.config.data_source, self._img_ids[index])

        image, image_meta = self._preprocess_image(self._read_image(image_path))
        return image, self._read_image(mask_path), image_meta

    def __len__(self):
        """ Returns size of the dataset """
        return len(self._img_ids)

    # Methods specific to the current implementation
    def _read_image(self, data_id):
        nib_image = nib.load(str(os.path.join(self.config.data_source, data_id)))
        image = np.array(nib_image.dataobj)
        if len(image.shape) != 4:  # Make sure 4D
            image = np.expand_dims(image, -1)
        image = np.transpose(image, (3, 0, 1, 2))

        return image

    def _preprocess_image(self, image):
        image_meta = {'image_shape': image.shape}

        # Swap modalities (mri_sequence)
        image = image[self.config.modality_order, :, :, :]
        # Crop
        image, bbox = self.crop(image)
        # Normalize
        image = self.normalize_img(image)
        # Resize
        shape = (image.shape[0],) + self.config.size
        image = resize3d(image, shape)

        image_meta['bbox'] = bbox

        return image, image_meta

    @staticmethod
    def crop(image):
        def bbox3d(img):
            nonzero_rows = np.any(img, axis=(1, 2)).nonzero()[0]
            nonzero_cols = np.any(img, axis=(0, 2)).nonzero()[0]
            nonzero_slices = np.any(img, axis=(0, 1)).nonzero()[0]

            bbox_ = np.array([[-1, -1, -1], [0, 0, 0]])
            if nonzero_rows.size > 0:
                bbox_[:, 0] = nonzero_rows[[0, -1]]
                bbox_[:, 1] = nonzero_cols[[0, -1]]
                bbox_[:, 2] = nonzero_slices[[0, -1]]

            return bbox_

        bboxes = np.stack([bbox3d(i) for i in image])
        bbox = np.stack([np.min(bboxes[:, 0, :], axis=0), np.max(bboxes[:, 1, :], axis=0)])

        image = image[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]]

        return image, bbox

    @staticmethod
    def normalize_img(image):
        for channel in range(image.shape[0]):
            img = image[channel, :, :, :].copy()
            mask = img > 0
            image_masked = np.ma.masked_array(img, ~mask)
            mean, std = np.mean(image_masked), np.std(image_masked)

            img -= mean
            img /= std
            image[channel, :, :, :] = img

        return image


# Custom implementation of Dice Index metric.
class DiceIndex(Metric):

    # Required methods
    def __init__(self, num_classes):
        self._classes_num = num_classes
        super().__init__()
        self._name = 'dice_index'
        self._overall_metric = []

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs.
        Possible format: {metric_name: metric_value}
        """
        return {self._name: np.mean(self._overall_metric)}

    def update(self, output, target):
        """ Calculates and updates metric value
        :param output: model output
        :param target: annotations
        """
        if len(output) != 1 or len(target) != 1:
            raise Exception('The Dice Index metric cannot be calculated '
                            'for a model with multiple outputs')

        output = output[0]
        target = target[0]
        result = np.zeros(shape=self._classes_num)
        for i in range(1, self._classes_num):
            annotation_data_ = (target == i)
            prediction_data_ = (output == i)

            intersection_count = np.logical_and(annotation_data_, prediction_data_).sum()
            union_count = annotation_data_.sum() + prediction_data_.sum()
            if union_count > 0:
                result[i] += 2.0*intersection_count / union_count

        annotation = (target > 0)
        prediction = (output > 0)

        intersection_count = np.logical_and(annotation, prediction).sum()
        union_count = annotation.sum() + prediction.sum()
        if union_count > 0:
            result[0] += 2.0 * intersection_count / union_count

        self._overall_metric.append(result)

    def reset(self):
        """ Resets metric """
        self._overall_metric = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'dice_index'}}


# Custom wrapper over IEEngine that implements postrocessor function, which can process the
# raw output of the model using metadata obtained during the image reading and preprocessing.
class SegmentationEngine(IEEngine):
    @staticmethod
    def postprocess_output(outputs, metadata):
        """
        Processes model raw output for future metric and loss calculation.
        Uses image metadata that can be passed using dataloader.
        :param outputs: network infer result in the format of dictionary numpy ndarray
                        by layer name (batch x image shape)
        :param metadata: dictionary of image metadata
        :return: processed numpy ndarray with the same shape as the original output
        """
        processed_outputs = []
        for output, meta in zip(outputs.values(), metadata):
            # Resize to bounding box size and extend to mask size
            output = output[0]
            low = meta['bbox'][0]
            high = meta['bbox'][1]
            box_shape = tuple((high - low).astype(np.int32))

            image_shape = meta['image_shape'][-3:]
            processed_output = np.zeros(shape=(output.shape[0],) + image_shape)

            processed_output[:, low[0]:high[0], low[1]:high[1], low[2]:high[2]] = \
                resize3d(output, shape=(output.shape[0],) + box_shape)

            # Transforms prediction from WT-TC-ET format to NCR/NET-ED-ET.
            # Elements passing the threshold of 0.5 fill with specified values
            result = np.zeros(shape=processed_output.shape[1:], dtype=np.int8) # pylint: disable=E1136

            label = processed_output > 0.5
            wt = label[0]
            tc = label[1]
            et = label[2]

            result[wt] = 1
            result[tc] = 2
            result[et] = 3

            processed_outputs.append(result)

        return np.stack(processed_outputs, axis=0)


def resize3d(image, shape):
    image = np.asarray(image)

    factor = [float(o) / i for i, o in zip(image.shape, shape)]
    image = interpolation.zoom(image, zoom=factor, order=1)

    return image


def main():
    parser = get_common_argparser()
    parser.add_argument(
        '--mask-dir',
        help='Path to the directory with segmentation masks',
        required=True
    )

    args = parser.parse_args()
    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    model_config = {
        'model_name': 'brain-tumor-segmentation-0002',
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
        'modality_order': [1, 2, 3, 0],
        'size': (128, 128, 128)
    }

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': 'ANY',
                'preset': 'performance',
                'stat_subset_size': 200
            }
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the data loader.
    data_loader = BRATSDataLoader(dataset_config)

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = DiceIndex(num_classes=4)

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = SegmentationEngine(config=engine_config,
                                data_loader=data_loader,
                                metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 6: Execute the pipeline.
    compressed_model = pipeline.run(model)

    # Step 7 (Optional):  Compress model weights to quantized precision
    #                     in order to reduce the size of final .bin file.
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
