# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from inspect import Parameter
import os
from sqlite3.dbapi2 import _Parameters
import numpy as np
from cv2 import imread, resize as cv2_resize

from openvino.tools.pot import quantize_post_training, QuantizationParameters, export
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser

# Custom DataLoader class implementation that is required for
# the proper reading of Imagenet images and annotations.
class ImageNetDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)
        self._annotations, self._img_ids = self._read_img_ids_annotations(self.config)

    def __len__(self):
        return len(self._img_ids)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        annotation = self._annotations[self._img_ids[index]] if self._annotations else None
        return self._read_image(self._img_ids[index]), annotation

    # Methods specific to the current implementation
    @staticmethod
    def _read_img_ids_annotations(dataset):
        """ Parses annotation file or directory with images to collect image names and annotations.
        :param dataset: dataset config
        :returns dictionary with annotations
                 list of image ids
        """
        annotations = {}
        img_ids = []
        if dataset.annotation_file:
            with open(dataset.annotation_file) as f:
                for line in f:
                    img_id, annotation = line.split(" ")
                    annotation = int(annotation.rstrip('\n'))
                    annotations[img_id] = annotation + 1 if dataset.has_background else annotation
                    img_ids.append(img_id)
        else:
            img_ids = sorted(os.listdir(dataset.data_source))

        return annotations, img_ids

    def _read_image(self, index):
        """ Reads images from directory.
        :param index: image index to read
        :return ndarray representation of image batch
        """
        image = imread(os.path.join(self.config.data_source, index))
        image = self._preprocess(image)
        return image

    def _preprocess(self, image):
        """ Does preprocessing of an image according to the preprocessing config.
        :param image: ndarray image
        :return processed image
        """
        for prep_params in self.config.preprocessing:
            image = PREPROC_FNS[prep_params.type](image, prep_params)
        return image

def resize(image, params):
    shape = params['height'], params['width']
    return cv2_resize(image, shape)


def crop(image, params):

    height, width = image.shape[:2]

    dst_height = int(height * params['central_fraction'])
    dst_width = int(width * params['central_fraction'])

    if height < dst_height or width < dst_width:
        resized = np.array([width, height])
        if width < dst_width:
            resized *= dst_width / width
        if height < dst_height:
            resized *= dst_height / height
        image = cv2_resize(image, tuple(np.ceil(resized).astype(int)))

    top_left_y = (height - dst_height) // 2
    top_left_x = (width - dst_width) // 2
    return image[top_left_y:top_left_y + dst_height, top_left_x:top_left_x + dst_width]


PREPROC_FNS = {'resize': resize, 'crop': crop}



def main():
    argparser = get_common_argparser()
    argparser.add_argument(
        '-a',
        '--annotation-file',
        help='File with Imagenet annotations in .txt format',
        required=True
    )

    parameters = QuantizationParameters()

    args = argparser.parse_args()

    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    model_config = {
        'model_name': 'sample_model',
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


    # Step 8: Save the compressed model to the desired path.
    save_model(compressed_model, os.path.join(os.path.curdir, 'optimized'))

    # Step 9 (Optional): Evaluate the compressed model. Print the results.
    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        for name, value in metric_results.items():
            print('{: <27s}: {}'.format(name, value))


if __name__ == '__main__':
    main()
