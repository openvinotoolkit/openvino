# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .provider import ClassProvider
import cv2
from random import randint
import numpy as np
import logging as log
import sys

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class CVFlip(ClassProvider):
    """
    Flip an input image using OpenCV. Suitable for 2D and 3D input data
    """
    __action_name__ = "cv2_flip"

    def __init__(self, config):
        """
        :param config: dictionary with transformer configuration.
        Mandatory config case:
        should have either 'random_flip_mode' key - in this case OpenCV flipCode will be randomly selected
        or have 'flip_mode' config key - allowed values 0 - horizontal flip, 1 - vertical flip, -1 - both flip
        Optional config keys:
        'target_layers' - defines for which layer from input data to apply transformation
        """
        if config.get("random_flip_mode", False):
            self.flip_mode = randint(-1, 1)
        else:
            self.flip_mode = config.get("flip_mode", 0)

        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Applying {} transformer for layers {} with flip mode {}...".format(self.__action_name__,
                                                                                     ", ".join(apply_to),
                                                                                     self.flip_mode))
        for layer in apply_to:
            try:
                assert len(data[layer].shape) == 3 or len(data[layer].shape) == 2, \
                    "Only 3D and 2D input data can be flipped"
                data[layer] = cv2.flip(src=data[layer], flipCode=self.flip_mode)
            except:
                log.error("Failed to process data for layer {}! Data processing skipped".format(layer))
                continue
        return data


class NumpyFlip(ClassProvider):
    """
    Flip an input image using numpy. Suitable for the data with arbitrary shape
    """
    __action_name__ = "np_flip"

    def __init__(self, config):
        """
        :param config: dictionary with transformer configuration.
        Optional config keys:
        'axis' - defines the axis in ndarray along which to flip a data. If not defined flipping will be perfromed
                 along all ndarray axises,
        'target_layers' - optional config key which defines to which layer from input data to apply transformation
        """
        self.axis = config.get("axis")
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Applying {} transformer for layers {} and axis {}...".format(self.__action_name__,
                                                                               ", ".join(apply_to),
                                                                               self.axis))
        for layer in apply_to:
            data[layer] = np.flip(data[layer], axis=self.axis)
        return data


class Crop(ClassProvider):
    """
    Crop an input image. Suitable only for 3D and 2D input data
    """
    __action_name__ = "crop"

    def __init__(self, config):
        """
        :param config: dictionary with transformer configuration.
        Mandatory config case:
        should have either 'random_crop' key - in this case crop range will be defined randomly with guarantee that 20%
                                               of pixels along each side will be kept
        or have 'x_crop' and 'y_crop' config key - tuples containing two int values defining crop region along x/y axis
        Optional config keys:
        'target_layers' - defines for which layer from input data to apply transformation
        """
        self.random_crop = config.get("random_crop", False)
        self.restore_initial_size = config.get("restore_initial_size", True)
        if not self.random_crop:
            self.x_crop = config['x_crop']
            self.y_crop = config['y_crop']
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()

        for layer in apply_to:
            try:
                assert len(data[layer].shape) == 3 or len(data[layer].shape) == 2, \
                    "Only 3D and 2D input data can be cropped"
                h, w, _ = data[layer].shape
                if self.random_crop:
                    # h or w // 10  required to guarantee that we will not crop whole image
                    # and keep at least 20 % of pixels along each side
                    x_start = randint(0, w // 2 - w // 10)
                    x_end = randint(w // 2 + w // 10, w)
                    self.x_crop = (x_start, x_end)
                    y_start = randint(0, h // 2 - h // 10)
                    y_end = randint(h // 2 + h // 10, h)
                    self.y_crop = (y_start, y_end)
                log.info("Crop data for layer {}. Crop region is: y crop range {}, x crop range {}...".format(layer,
                                                                                                              self.y_crop,
                                                                                                              self.x_crop))
                data[layer] = data[layer][self.y_crop[0]:self.y_crop[1], self.x_crop[0]:self.x_crop[1]]
                if self.restore_initial_size:
                    data[layer] = cv2.resize(data[layer], (h, w))
            except:
                log.error("Failed to process data for layer {}! Data processing skipped".format(layer))
                continue
        return data


class InvertData(ClassProvider):
    """
    Invert input data relatively to array max value. After transformation each n-th array element
    will have value equal to (array_max_value - n-th ndarray element)
    """
    __action_name__ = "invert_data"

    def __init__(self, config):
        """
        :param config: dictionary with transformer configuration.
        'target_layers' - defines for which layer from input data to apply transformation
        """
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Invert colors for layer {}...".format(", ".join(apply_to)))
        for layer in apply_to:
            data[layer] = np.full(fill_value=data[layer].max, shape=data[layer].shape) - data[layer]
        return data


class AddGaussianNoise(ClassProvider):
    """
    Adds random data with gaussian distribution to an input data
    """
    __action_name__ = "add_gaussian_noise"

    def __init__(self, config):
        """
        :param config: dictionary with transformer configuration.
        Mandatory config case:
        should have either 'auto_range' key - in this case mean value and standard deviation will be calculated
                                              automatically basing on input data range
        or have 'mean' and 'sigma' config key - numeric values for mean and standard deviation accordingly
        Optional config keys:
        'target_layers' - defines for which layer from input data to apply transformation
        """

        self.target_layers = config.get('target_layers', None)
        self.auto_range = config.get("auto_range", False)
        if not self.auto_range:
            self.mean = config.get("mean", 0)
            self.sigma = config.get("sigma", 0.01)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        for layer in apply_to:
            if self.auto_range:
                self.mean = np.mean(data[layer]) / 5
                self.sigma = np.std(data[layer]) / 10
            log.info("Add gaussian noise for input data for layer {} with mean={} and std={}".format(layer,
                                                                                                     self.mean,
                                                                                                     self.sigma))
            noise = np.random.normal(loc=self.mean, scale=self.sigma, size=data[layer].shape)
            data[layer] = np.clip((data[layer] + noise), a_min=0, a_max=255).astype(np.uint8)
        return data
