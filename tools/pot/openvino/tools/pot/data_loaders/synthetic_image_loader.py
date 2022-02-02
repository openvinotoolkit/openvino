# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing import Pool
from pathlib import Path
import os
import re
import requests

import cv2 as cv
import numpy as np

from openvino.runtime import Layout # pylint: disable=E0611,E0401
from openvino.tools.pot.utils.logger import get_logger
from openvino.tools.pot.data_loaders.image_loader import ImageLoader
from .utils import collect_img_files

logger = get_logger(__name__)


class IFSFunction:
    def __init__(self, prev_x, prev_y):
        self.function = []
        self.xs, self.ys = [prev_x], [prev_y]
        self.select_function = []
        self.cum_proba = 0.0

    def set_param(self, params, proba, weights=None):
        if weights is not None:
            params = list(np.array(params) * np.array(weights))

        self.function.append(params)
        self.cum_proba += proba
        self.select_function.append(self.cum_proba)

    def calculate(self, iteration):
        rand = np.random.random(iteration)
        prev_x, prev_y = 0, 0
        next_x, next_y = 0, 0

        for i in range(iteration):
            for func_params, select_func in zip(self.function, self.select_function):
                a, b, c, d, e, f = func_params
                if rand[i] <= select_func:
                    next_x = prev_x * a + prev_y * b + e
                    next_y = prev_x * c + prev_y * d + f
                    break

            self.xs.append(next_x)
            self.ys.append(next_y)
            prev_x = next_x
            prev_y = next_y

    @staticmethod
    def process_nans(data):
        nan_index = np.where(np.isnan(data))
        extend = np.array(range(nan_index[0][0] - 100, nan_index[0][0]))
        delete_row = np.append(extend, nan_index)
        return delete_row

    def rescale(self, image_x, image_y, pad_x, pad_y):
        xs = np.array(self.xs)
        ys = np.array(self.ys)
        if np.any(np.isnan(xs)):
            delete_row = self.process_nans(xs)
            xs = np.delete(xs, delete_row, axis=0)
            ys = np.delete(ys, delete_row, axis=0)

        if np.any(np.isnan(ys)):
            delete_row = self.process_nans(ys)
            xs = np.delete(xs, delete_row, axis=0)
            ys = np.delete(ys, delete_row, axis=0)

        if np.min(xs) < 0.0:
            xs -= np.min(xs)
        if np.min(ys) < 0.0:
            ys -= np.min(ys)
        xmax, xmin = np.max(xs), np.min(xs)
        ymax, ymin = np.max(ys), np.min(ys)
        self.xs = np.uint16(xs / (xmax - xmin) * (image_x - 2 * pad_x) + pad_x)
        self.ys = np.uint16(ys / (ymax - ymin) * (image_y - 2 * pad_y) + pad_y)

    def draw(self, draw_type, image_x, image_y, pad_x=6, pad_y=6):
        self.rescale(image_x, image_y, pad_x, pad_y)
        image = np.zeros((image_x, image_y), dtype=np.uint8)

        for i in range(len(self.xs)):
            if draw_type == 'point':
                image[self.ys[i], self.xs[i]] = 127
            else:
                mask = '{:09b}'.format(np.random.randint(1, 512))
                patch = 127 * np.array(list(map(int, list(mask))), dtype=np.uint8).reshape(3, 3)
                x_start = self.xs[i] + 1
                y_start = self.ys[i] + 1
                image[x_start:x_start+3, y_start:y_start+3] = patch

        return image


class SyntheticImageLoader(ImageLoader):
    def __init__(self, config):
        super().__init__(config)

        np.random.seed(seed=1)
        self.subset_size = config.get('subset_size', 300)
        self._cpu_count = min(os.cpu_count(), self.subset_size)
        self._shape = config.get('shape', None)
        self.data_source = config.get('data_source', None)
        self._weights = np.array([
            0.2, 1, 1, 1, 1, 1,
            0.6, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
            1.4, 1, 1, 1, 1, 1,
            1.8, 1, 1, 1, 1, 1,
            1, 0.2, 1, 1, 1, 1,
            1, 0.6, 1, 1, 1, 1,
            1, 1.4, 1, 1, 1, 1,
            1, 1.8, 1, 1, 1, 1,
            1, 1, 0.2, 1, 1, 1,
            1, 1, 0.6, 1, 1, 1,
            1, 1, 1.4, 1, 1, 1,
            1, 1, 1.8, 1, 1, 1,
            1, 1, 1, 0.2, 1, 1,
            1, 1, 1, 0.6, 1, 1,
            1, 1, 1, 1.4, 1, 1,
            1, 1, 1, 1.8, 1, 1,
            1, 1, 1, 1, 0.2, 1,
            1, 1, 1, 1, 0.6, 1,
            1, 1, 1, 1, 1.4, 1,
            1, 1, 1, 1, 1.8, 1,
            1, 1, 1, 1, 1, 0.2,
            1, 1, 1, 1, 1, 0.6,
            1, 1, 1, 1, 1, 1.4,
            1, 1, 1, 1, 1, 1.8,
        ]).reshape(-1, 6)
        self._threshold = 0.2
        self._iterations = 200000
        self._num_of_points = None
        self._instances = None
        self._categories = None
        if isinstance(self._shape, str):
            self._shape = list(map(int, re.findall(r'\d+', self._shape)))

        super().get_layout()
        self._check_input_shape()

        if os.path.exists(self.data_source) and os.listdir(self.data_source) and not config.generate_data:
            logger.info(f'Dataset was found in `{self.data_source}`')
        else:
            logger.info(f'Synthetic dataset will be stored in `{self.data_source}`')
            if not os.path.exists(self.data_source):
                os.mkdir(self.data_source)

        assert os.path.isdir(self.data_source)
        if config.generate_data or not os.listdir(self.data_source):
            self._download_colorization_model()
            logger.info(f'Start generating {self.subset_size} synthetic images')
            self.generate_dataset()

        self._img_files = collect_img_files(self.data_source)

    def _check_input_shape(self):
        if self._shape is None:
            raise ValueError('Input shape should be specified. Please, use `--shape`')
        if  len(self._shape) < 3 or len(self._shape) > 4:
            raise ValueError(f'Input shape should have 3 or 4 dimensions, but provided {self._shape}')
        if self._shape[self._layout.get_index_by_name('C')] != 3:
            raise ValueError('SyntheticImageLoader can generate images with only channels == 3')

    def _download_colorization_model(self):
        proto_name = 'colorization_deploy_v2.prototxt'
        model_name = 'colorization_release_v2.caffemodel'
        npy_name = 'pts_in_hull.npy'

        if not os.path.exists(proto_name):
            url = 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/'
            proto = requests.get(url + proto_name)
            open(proto_name, 'wb').write(proto.content)
        if not os.path.exists(model_name):
            url = 'http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/'
            model = requests.get(url + model_name)
            open(model_name, 'wb').write(model.content)
        if not os.path.exists(npy_name):
            url = 'https://github.com/richzhang/colorization/raw/caffe/colorization/resources/'
            pts_in_hull = requests.get(url + npy_name)
            open(npy_name, 'wb').write(pts_in_hull.content)

    def _initialize_params(self, height, width):
        default_img_size = 362 * 362
        points_coeff = max(1, int(np.round(height * width / default_img_size)))
        self._num_of_points = 100000 * points_coeff

        if self.subset_size < len(self._weights):
            self._instances = 1
            self._categories = 1
            self._weights = self._weights[:self.subset_size, :]
        else:
            self._instances = np.ceil(0.25 * self.subset_size / self._weights.shape[0]).astype(int)
            self._categories = np.ceil(self.subset_size / (self._instances * self._weights.shape[0])).astype(int)

    def generate_dataset(self):
        height = self._shape[self._layout.get_index_by_name('H')]
        width = self._shape[self._layout.get_index_by_name('W')]
        self._initialize_params(height, width)

        # to avoid multiprocessing error: can't pickle openvino.pyopenvino.Layout objects
        self._layout = str(self._layout)

        with Pool(processes=self._cpu_count) as pool:
            params = pool.map(self._generate_category, [1e-5] * self._categories)

        instances_weights = np.repeat(self._weights, self._instances, axis=0)
        weight_per_img = np.tile(instances_weights, (self._categories, 1))
        repeated_params = np.repeat(params, self._weights.shape[0] * self._instances, axis=0)
        repeated_params = repeated_params[:self.subset_size]
        weight_per_img = weight_per_img[:self.subset_size]
        assert weight_per_img.shape[0] == len(repeated_params) == self.subset_size

        splits = min(self._cpu_count, self.subset_size)
        params_per_proc = np.array_split(repeated_params, splits)
        weights_per_proc = np.array_split(weight_per_img, splits)

        generation_params = []
        offset = 0
        for param, w in zip(params_per_proc, weights_per_proc):
            indices = list(range(offset, offset + len(param)))
            offset += len(param)
            generation_params.append((param, w, height, width, indices))

        with Pool(processes=self._cpu_count) as pool:
            pool.starmap(self._generate_image_batch, generation_params)

        self._layout = Layout(self._layout)

    def _generate_image_batch(self, params, weights, height, width, indices):
        pts_in_hull = np.load('pts_in_hull.npy').transpose().reshape(2, 313, 1, 1).astype(np.float32)
        net = cv.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
        net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
        net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        for i, param, weight in zip(indices, params, weights):
            image = self._generator(param, 'gray', self._iterations, height, width, weight)
            color_image = self._colorize(image, net)
            aug_image = self._augment(color_image)
            cv.imwrite(os.path.join(self.data_source, "{:06d}.png".format(i)), aug_image)

    @staticmethod
    def _generator(params, draw_type, iterations, height=512, width=512, weight=None):
        generators = IFSFunction(prev_x=0.0, prev_y=0.0)
        for param in params:
            generators.set_param(param[:6], param[6], weight)
        generators.calculate(iterations)
        img = generators.draw(draw_type, height, width)
        return img

    def _generate_category(self, eps, height=512, width=512):
        pixels = -1
        while pixels < self._threshold:
            param_size = np.random.randint(2, 8)
            params = np.zeros((param_size, 7), dtype=np.float32)

            sum_proba = eps
            for i in range(param_size):
                a, b, c, d, e, f = np.random.uniform(-1.0, 1.0, 6)
                prob = abs(a * d - b * c)
                sum_proba += prob
                params[i] = a, b, c, d, e, f, prob
            params[:, 6] /= sum_proba

            fracral_img = self._generator(params, 'point', self._num_of_points, height, width)
            pixels = np.count_nonzero(fracral_img) / (height * width)
        return params

    @staticmethod
    def _rgb2lab(frame):
        y_coeffs = np.array([0.212671, 0.715160, 0.072169], dtype=np.float32)
        frame = np.where(frame > 0.04045, np.power((frame + 0.055) / 1.055, 2.4), frame / 12.92)
        y = frame @ y_coeffs.T
        L = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
        return L

    def _colorize(self, frame, net):
        H_orig, W_orig = frame.shape[:2] # original image size
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = np.tile(frame.reshape(H_orig, W_orig, 1), (1, 1, 3))

        frame = frame.astype(np.float32) / 255
        img_l = self._rgb2lab(frame) # get L from Lab image
        img_rs = cv.resize(img_l, (224, 224)) # resize image to network input size
        img_l_rs = img_rs - 50  # subtract 50 for mean-centering

        net.setInput(cv.dnn.blobFromImage(img_l_rs))
        ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
        img_lab_out = np.concatenate((img_l[..., np.newaxis], ab_dec_us), axis=2) # concatenate with original image L
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)
        frame_normed = 255 * (img_bgr_out - img_bgr_out.min()) / (img_bgr_out.max() - img_bgr_out.min())
        frame_normed = np.array(frame_normed, dtype=np.uint8)
        return cv.resize(frame_normed, (H_orig, W_orig))

    def _augment(self, image):
        if np.random.random(1) >= 0.5:
            image = cv.flip(image, 1)

        if np.random.random(1) >= 0.5:
            image = cv.flip(image, 0)

        height, width = image.shape[:2]
        angle = np.random.uniform(-30, 30)
        rotate_matrix = cv.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=1)
        image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

        image = self._fill_background(image)

        k_size = np.random.choice(list(range(1, 16, 2)))
        image = cv.GaussianBlur(image, (k_size, k_size), 0)
        return image

    @staticmethod
    def _fill_background(image):
        synthetic_background = Path(__file__).parent / 'synthetic_background.npy'
        imagenet_means = np.load(synthetic_background)
        class_id = np.random.randint(0, imagenet_means.shape[0])
        rows, cols = np.where(~np.any(image, axis=-1))  # background color = [0, 0, 0]
        image[rows, cols] = imagenet_means[class_id]
        return image
