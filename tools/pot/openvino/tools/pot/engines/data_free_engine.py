# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import cv2 as cv
import requests
import tarfile
import urllib

from copy import deepcopy
from joblib import Parallel, delayed
from pathlib import Path

from openvino.runtime import Layout  # pylint: disable=E0611,E0401
from openvino.tools.pot.openvino.tools.pot.data_loaders.image_loader import ImageLoader
from openvino.tools.pot.openvino.tools.pot.utils import logger
from .simplified_engine import SimplifiedEngine
from openvino.tools.pot.graph.model_utils import get_nodes_by_type

class ifs_function():
    def __init__(self, prev_x, prev_y):
        self.function = []
        self.xs, self.ys = [prev_x], [prev_y]
        self.select_function = []
        self.cum_proba = 0.0

    def set_param(self, params, proba, weights=None):
        if weights:
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
        image = np.empty((image_x, image_y, 3 if draw_type == 'color' else 1))

        for i in range(len(self.xs)):
            mask_pattern = '{:09b}'.format(np.random.randint(1, 512))
            if draw_type == 'color':
                patch = self.make_patch3_3(mask_pattern, [self.convert_color(i, 128)])
            elif draw_type == 'point':
                patch = np.array([127, 127, 127])
            else:
                patch = self.make_patch3_3(mask_pattern, [127, 127, 127])

            # why + 1????
            if draw_type == 'point':
                image = np.array(image)
                image[self.ys[i] + 1, self.xs[i] + 1, :] = 127, 127, 127
                # image.paste(patch, (self.xs[i], self.ys[i]))
            else:
                H, W, C = patch.shape
                x_start = self.xs[i] + 1
                y_start = self.ys[i] + 1
                image[x_start : x_start + H, y_start : y_start + W, :] = patch
                # Image.fromarray(patch)
                # image.paste(patch, (self.xs[i] + 1, self.ys[i] + 1))

        image = np.array(image, dtype='uint8')
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image

    def make_patch3_3(self, mask, patch_color):
        patch_color = np.array(patch_color)
        patch = np.zeros((3, 3, 3), np.uint8)
        for i in range(3):
            for j in range(3):
                patch[i ,j, :] = patch_color * int(mask[i*3+j])
        return patch


class DataFreeEngine(SimplifiedEngine):

    def __init__(self, config, stat_subset_size=300):
        super().__init__(config)
        np.random.seed(seed=42)
        self._cpu_count = os.cpu_count()

        self.shape = config.get('shape', None)
        self._layout = config.get('layout', None)
        if not self.shape:
            inputs = get_nodes_by_type(self.model, ['Parameter'], recursively=False)
            if len(inputs) > 1:
                raise RuntimeError('DataFreeEngine supports networks with single image input'
                                   'Actual inputs number: {}'.format(len(inputs)))
            self.shape = inputs[0].shape
            self.data_loader = self.get_data_loader(input_node=inputs[0])
        else:
            self.data_loader = self.get_data_loader()

        self.download = config.get('download', True)
        self.dataset_dir = config.get('dataset_dir', Path(__file__).resolve().parents[7])
        logger.info(f'Syntetic dataset will be stored in {self.dataset_dir}')
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

            if isinstance(self._data_loader, ImageLoader):
                if self.download:
                    raise NotImplementedError
                    # URL = None
                    # response  = requests.get(URL, stream=True)
                    # archive_path = os.path.join(self.dataset_dir, 'FractalDB-1k.tar.gz')
                    # with open(archive_path, 'wb') as f:
                    #     f.write(response.raw.read())

                    # tar = tarfile.open(archive_path, "r:gz")
                    # tar.extractall(path=archive_path.replace('.tar.gz', ''))
                    # tar.close()
                else:
                    # progress bar?
                    self._threshold = 0.2
                    self._iterations = 200000

                    height, width = self.shape[-2], self.shape[-1]
                    default_img_size = 362 * 362
                    points_coeff = np.max(1, np.round(height * width / default_img_size))
                    self._num_of_points = 100000 * points_coeff

                    if stat_subset_size < len(self._weights):
                        self._instances = 1
                        self._categories = 1
                        self._weights = self._weights[:stat_subset_size]
                    else:
                        self._instances = np.ceil(0.25 * stat_subset_size / len(self._weights))
                        self._categories = np.ceil(stat_subset_size / (self._instances * len(self._weights)))

                    self.generate_dataset()

        if isinstance(self._data_loader, ImageLoader) and self._shape[self._layout.get_index_by_name('C')] == 3:
            # download colorization model files
            self._net = cv.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
            pts_in_hull = np.load('pts_in_hull.npy')
            pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
            self._net.getLayer(self._net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
            self._net.getLayer(self._net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    def get_data_loader(self, input_node=None):
        num_dims = len(self._shape)

        if self._layout is None and input_node is not None:
            layout_from_ir = input_node.graph.graph.get('layout', None)
            if layout_from_ir is not None:
                self._layout = Layout(layout_from_ir)

        if num_dims == 2:
            if self._layout is None:
                self._layout = Layout('NC')
            raise NotImplementedError
            # return TextLoader()

        if num_dims == 3:
            if self._layout is not None and 'C' in self._layout and 'H' in self._layout and 'W' in self._layout:
                return ImageLoader(shape=self.shape)
            if self._layout is None or self._layout == Layout('ANY'):
                if self._shape[0] in [1, 3]:
                    self._layout = Layout("CHW")
                    return ImageLoader(shape=self.shape)
                if self._shape[-1] in [1, 3]:
                    self._layout = Layout("HWC")
                    return ImageLoader(shape=self.shape)

                raise NotImplementedError
                # return AudioLoader(shape=self.shape)  # rnnt

        if num_dims == 4:
            if self._layout is not None and 'C' in self._layout and 'H' in self._layout and 'W' in self._layout:
                return ImageLoader(shape=self.shape)

            if self._layout is None or self._layout == Layout('ANY'):
                if self._shape[1] in [1, 3]:
                    self._layout = Layout("NCHW")
                    return ImageLoader(shape=self.shape)
                if self._shape[-1] in [1, 3]:
                    self._layout = Layout("NHWC")
                    return ImageLoader(shape=self.shape)

                raise NotImplementedError
                # return AudioLoader(shape=self.shape)  # mozilla


    def generate_dataset(self):
        height, width = self.shape[-2], self.shape[-1]
        n_jobs = min(self._cat_num, self._cpu_count)
        params = Parallel(n_jobs=n_jobs)(delayed(self._generate_category)() for _ in range(self._cat_num))

        instances_weights = np.repeat(self._weights, self._instances)
        weight_per_img = np.tile(instances_weights, len(params))
        repeated_params = np.repeat(params, len(self._weights) * len(self._instances))
        assert len(weight_per_img) == len(repeated_params)

        Parallel(n_jobs=self._cpu_count)(delayed(self.generate_image)(param, w, height, width, i)
                                   for i, param, w in enumerate(zip(repeated_params, weight_per_img)))

    def generate_image(self, param, weight, height, width, i):
        image = self._generator(param, 'gray', self._iterations, height, width, weight)
        color_image = self._colorize(image)
        aug_image = self._augment(color_image)
        cv.imwrite(os.path.join(self.dataset_dir, "{:06d}.png".format(i)), aug_image)

    @staticmethod
    def _generator(params, draw_type, iterations, height=512, width=512, weight=None):
        generators = ifs_function(prev_x=0.0, prev_y=0.0)
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
            pixels = np.count_nonzero(fracral_img[:,:,0]) / (height * width)

        return params

    def _colorize(self, frame):
        # -----------preprocessing-----------
        img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
        (H_orig, W_orig) = img_rgb.shape[:2] # original image size
        img_rs = cv.resize(img_rgb, (224, 224)) # resize image to network input size
        img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
        img_l_rs = img_lab_rs[:,:,0]
        img_l_rs -= 50 # subtract 50 for mean-centering

        # -----------run network-----------
        net = deepcopy(self._net)
        net.setInput(cv.dnn.blobFromImage(img_l_rs))
        ab_dec = net.forward()[0,:,:,:].transpose((1,2,0))

        # -----------postprocessing-----------
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
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

        k_size = np.random.randint(1, 16)
        image = cv.GaussianBlur(image, (k_size, k_size), 0)

        height, width, _ = image.shape
        angle = np.random.uniform(-30, 30)
        rotate_matrix = cv.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=1)
        image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        return image