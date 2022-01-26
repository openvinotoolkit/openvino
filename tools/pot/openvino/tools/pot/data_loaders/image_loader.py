# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from cv2 import imread, IMREAD_GRAYSCALE

from openvino.runtime import Layout, Dimension # pylint: disable=E0611,E0401
from ..api.data_loader import DataLoader
from ..data_loaders.utils import prepare_image, collect_img_files


class ImageLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)

        self._img_files = collect_img_files(config.data_source)
        self._shape = None
        self._layout = config.get('layout', None)
        self._crop_central_fraction = config.get('central_fraction', None)

    def __getitem__(self, idx):
        return self._read_and_preproc_image(self._img_files[idx])

    def __len__(self):
        return len(self._img_files)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    def _read_and_preproc_image(self, img_path):
        image = imread(img_path, IMREAD_GRAYSCALE)\
            if self._shape[1] == 1 else imread(img_path)

        if image is None:
            raise Exception('Can not read the image: {}'.format(img_path))

        return prepare_image(image, self._layout, self.shape[-2:], self._crop_central_fraction)

    def get_layout(self, input_node):
        if self._layout is not None:
            if 'C' not in self._layout or 'H' not in self._layout or 'W' not in self._layout:
                raise ValueError('Unexpected {} layout'.format(self._layout))
            self._layout = Layout(self._layout)
            return

        layout_from_ir = input_node.graph.graph.get('layout', None)
        if layout_from_ir is not None:
            self._layout = Layout(layout_from_ir)
            return

        image_colors_dim = (Dimension(3), Dimension(1))
        num_dims = len(self._shape)
        if num_dims == 4:
            if self._shape[1] in image_colors_dim:
                self._layout = Layout("NCHW")
            elif self._shape[3] in image_colors_dim:
                self._layout = Layout("NHWC")
        elif num_dims == 3:
            if self._shape[0] in image_colors_dim:
                self._layout = Layout("CHW")
            elif self._shape[2] in image_colors_dim:
                self._layout = Layout("HWC")
