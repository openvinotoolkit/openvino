# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import cv2 as cv
import requests
import tarfile

from copy import deepcopy
from joblib import Parallel, delayed

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
        image = np.empty((image_x, image_y))

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

    def make_patch3_3(self, mask, patch_color):
        patch_color = np.array(patch_color)
        patch = np.zeros((3, 3, 3), np.uint8)
        for i in range(3):
            for j in range(3):
                patch[i ,j, :] = patch_color * int(mask[i*3+j])
        return patch


class SyntheticImageLoader(ImageLoader):
    def __init__(self, config):
        super().__init__(config)
        np.random.seed(seed=1)
        self._cpu_count = os.cpu_count()
        self._shape = config.input_shape
        self.data_source = config.data_source
        self.subset_size = config.subset_size
        self._weights = np.array([
            0.2, 1, 1, 1, 1, 1,
            0.6, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
            1.4, 1, 1, 1, 1, 1,
            1.8, 1, 1, 1, 1, 1,
            1, 0.2, 1, 1, 1,1,
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

        self._net = cv.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
        pts_in_hull = np.load('pts_in_hull.npy').transpose().reshape(2, 313, 1, 1)
        self._net.getLayer(self._net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
        self._net.getLayer(self._net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        if not os.path.exists(self.data_source):
            logger.info(f'synthetic dataset will be stored in {self.data_source}')
            os.mkdir(self.data_source)
        else:
            logger.info(f'Dataset was found in {self.data_source}')

        assert os.path.isdir(self.data_source)
        if not os.listdir(self.data_source): # folder is empty
            if config.generate_data:
                # progress bar?
                self.generate_dataset()
            else:
                # self.download_images()
                raise NotImplementedError

        self._img_files = collect_img_files(self.data_source)

    def download_images(self):
        URL = ''
        response  = requests.get(URL, stream=True)
        archive_path = os.path.join(self.data_source, 'synthetic_images.tar.gz')
        with open(archive_path, 'wb') as f:
            f.write(response.raw.read())

        tar = tarfile.open(archive_path, "r:gz")
        tar.extractall(path=archive_path.replace('.tar.gz', ''))
        tar.close()

    def initialize_params(self):
        self._threshold = 0.2
        self._iterations = 200000

        height, width = self._shape[-2], self._shape[-1]
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
        height, width = self._shape[-2], self._shape[-1]
        self.initialize_params()
        n_jobs = min(self._categories, self._cpu_count)
        params = Parallel(n_jobs=1)(delayed(self._generate_category)() for _ in range(self._categories))

        instances_weights = np.repeat(self._weights, self._instances, axis=0)
        weight_per_img = np.tile(instances_weights, (self._categories, 1))
        repeated_params = np.repeat(params, self._weights.shape[0] * self._instances)
        assert weight_per_img.shape[0] == len(repeated_params)

        # Parallel(n_jobs=self._cpu_count)(delayed(self.generate_image)(param, w, height, width, i)
        Parallel(n_jobs=1)(delayed(self.generate_image)(param, w, height, width, i)
                                   for i, (param, w) in enumerate(zip(repeated_params, weight_per_img)))

    def generate_image(self, param, weight, height, width, i):
        image = self._generator(param, 'gray', self._iterations, height, width, weight)
        color_image = self._colorize(image)
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

    def _generate_category(self, height=512, width=512):
        pixels = -1
        while pixels < self._threshold:
            param_size = np.random.randint(2, 8)
            params = np.zeros((param_size, 7), dtype=float)

            sum_proba = 1e-5
            for i in range(param_size):
                a, b, c, d, e, f = np.random.uniform(-1.0, 1.0, 6)
                prob = abs(a * d - b * c)
                sum_proba += prob
                params[i] = a, b, c, d, e, f, prob
            params[:, 6] /= sum_proba

            fracral_img = self._generator(params, 'point', self._num_of_points, height, width)
            pixels = np.count_nonzero(fracral_img) / (height * width)

        return params

    def _colorize(self, frame):
        # -----------preprocessing-----------
        H_orig, W_orig = frame.shape[:2] # original image size
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = np.tile(frame.reshape(H_orig, W_orig, 1), (1, 1, 3))
            # cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        frame = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(frame, cv.COLOR_BGR2Lab)
        img_rs = cv.resize(img_lab, (224, 224)) # resize image to network input size
        img_l_rs = img_rs[:,:,0] - 50  # subtract 50 for mean-centering

        # -----------run network-----------
        # net = deepcopy(self._net)
        net = self._net
        net.setInput(cv.dnn.blobFromImage(img_l_rs))
        ab_dec = net.forward()[0,:,:,:].transpose((1,2,0))

        # -----------postprocessing-----------
        img_l = img_lab[:,:,0] # pull out L channel
        ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
        img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)
        frame_normed = 255 * (img_bgr_out - img_bgr_out.min()) / (img_bgr_out.max() - img_bgr_out.min())
        frame_normed = np.array(frame_normed, dtype='uint8')
        return cv.resize(frame_normed, frame.shape[:2])

    @staticmethod
    def _augment(image):
        if np.random.random(1) >= 0.5:
            image = cv.flip(image, 1)

        if np.random.random(1) >= 0.5:
            image = cv.flip(image, 0)

        k_size = np.random.choice(list(range(1, 16, 2)))
        image = cv.GaussianBlur(image, (k_size, k_size), 0)

        height, width, _ = image.shape
        angle = np.random.uniform(-30, 30)
        rotate_matrix = cv.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=1)
        image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        return image
