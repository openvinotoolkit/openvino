# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from openvino import Layout, PartialShape, Type
from openvino.tools.benchmark.utils import inputs_filling
from openvino.tools.benchmark.utils.utils import AppInputInfo


class TestImageResize(unittest.TestCase):
    def load_image(self, image, resized, layout):
        info = AppInputInfo()
        info.name = "input"
        info.element_type = Type.u8
        info.layout = Layout(layout)
        info.partial_shape = PartialShape(
            [1, 3, 2, 4] if layout == "NCHW" else [1, 2, 4, 3]
        )
        image_io = Mock()
        image_io.imread.return_value = image
        image_io.resize.return_value = resized
        with (
            patch.object(inputs_filling, "cv2", image_io, create=True),
            patch.dict(sys.modules, {"cv2": image_io}),
        ):
            tensors = inputs_filling.get_image_tensors(["input.png"], info, [1])
        expected = resized.transpose(2, 0, 1) if layout == "NCHW" else resized
        np.testing.assert_array_equal(tensors[0].data[0], expected)
        return image_io

    def test_mismatched_image_dimensions_are_resized(self):
        resized = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        for layout in ["NCHW", "NHWC"]:
            for image_shape in [(4, 2, 3), (5, 6, 3)]:
                with self.subTest(layout=layout, image_shape=image_shape):
                    image = np.zeros(image_shape, dtype=np.uint8)
                    image_io = self.load_image(image, resized, layout)
                    image_io.resize.assert_called_once_with(image, (4, 2))

    def test_matching_rectangular_image_is_not_resized(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        for layout in ["NCHW", "NHWC"]:
            with self.subTest(layout=layout):
                image_io = self.load_image(image, image, layout)
                image_io.resize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
