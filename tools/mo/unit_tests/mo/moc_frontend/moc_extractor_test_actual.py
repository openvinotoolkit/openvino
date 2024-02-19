# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.moc_frontend.extractor import decode_name_with_port
from openvino.tools.mo.utils.error import Error

import pytest


mock_available = True

try:
    # pylint: disable=no-name-in-module,import-error
    from mock_mo_python_api import get_model_statistic, get_place_statistic, \
        clear_frontend_statistic, clear_model_statistic, clear_place_statistic, \
        clear_setup, set_equal_data, set_max_port_counts

    # pylint: disable=no-name-in-module,import-error
    from openvino.frontend import FrontEndManager

except Exception:
    print("No mock frontend API available, "
          "ensure to use -DENABLE_TESTS=ON option when running these tests")
    mock_available = False

# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins,
# no objects shall exist after this
if mock_available:
    fem = FrontEndManager()

mock_needed = pytest.mark.skipif(not mock_available,
                                 reason="mock MO fe is not available")


class TestMainFrontend(unittest.TestCase):
    def setUp(self):
        clear_frontend_statistic()
        clear_model_statistic()
        clear_place_statistic()
        clear_setup()
        set_max_port_counts(10, 10)
        self.fe = fem.load_by_framework('openvino_mock_mo_frontend')
        self.model = self.fe.load('abc.bin')

    # Mock model has 'tensor' tensor place
    @mock_needed
    def test_decode_name_with_port_tensor(self):
        node = decode_name_with_port(self.model, "tensor")
        model_stat = get_model_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert node

    # pylint: disable=wrong-spelling-in-comment
    # Mock model doesn't have 'mocknoname' place
    @mock_needed
    def test_decode_name_with_port_noname(self):
        with self.assertRaisesRegex(Error, 'No\\ node\\ with\\ name.*mocknoname*'):
            decode_name_with_port(self.model, 'mocknoname')
        model_stat = get_model_statistic()
        assert model_stat.get_place_by_tensor_name == 1

    # Mock model has both tensor and tensor:0 places with non equal data
    # Collision is expected
    @mock_needed
    def test_decode_name_with_port_collision(self):
        with self.assertRaisesRegex(Error, 'Name\\ collision.*tensorAndOp*'):
            decode_name_with_port(self.model, 'tensorAndOp:0')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.is_equal_data > 0


    # Mock model has 'operation' and output port up to 10
    @mock_needed
    def test_decode_name_with_port_delim_op_out(self):
        node = decode_name_with_port(self.model, 'operation:7')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_output_port == 1
        assert place_stat.lastArgInt == 7
        assert node

    # Mock model has 'operation' and input port up to 10
    @mock_needed
    def test_decode_name_with_port_delim_op_in(self):
        node = decode_name_with_port(self.model, '7:operation')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_input_port == 1
        assert place_stat.lastArgInt == 7
        assert node

    # Mock model has 'tensor' and 'tensor:0' tensor places, no collision is expected
    @mock_needed
    def test_decode_name_with_port_delim_tensor_no_collision_out(self):
        node = decode_name_with_port(self.model, 'tensor:0')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_output_port == 0
        assert node

    # Mock model has 'tensor' and '0:tensor' tensor places, no collision is expected
    @mock_needed
    def test_decode_name_with_port_delim_tensor_no_collision_in(self):
        node = decode_name_with_port(self.model, '0:tensor')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_input_port == 0
        assert node

    # Mock model doesn't have such '1234:operation' or output port=1234 for 'operation'
    @mock_needed
    def test_decode_name_with_port_delim_no_port_out(self):
        with self.assertRaisesRegex(Error, 'No\\ node\\ with\\ name.*operation\\:1234*'):
            decode_name_with_port(self.model, 'operation:1234')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_output_port == 1
        assert place_stat.lastArgInt == 1234

    # Mock model doesn't have such '1234:operation' or input port=1234 for 'operation'
    @mock_needed
    def test_decode_name_with_port_delim_no_port_in(self):
        with self.assertRaisesRegex(Error, 'No\\ node\\ with\\ name.*1234\\:operation*'):
            decode_name_with_port(self.model, '1234:operation')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_input_port == 1
        assert place_stat.lastArgInt == 1234

    # Mock model has tensor with name 'conv2d:0' and operation 'conv2d' with output port = 1
    # It is setup to return 'is_equal_data=True' for these tensor and port
    # So no collision is expected
    @mock_needed
    def test_decode_name_with_port_delim_equal_data_out(self):
        set_equal_data('conv2d', 'conv2d')
        node = decode_name_with_port(self.model, 'conv2d:0')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_output_port == 1
        assert place_stat.is_equal_data > 0
        assert node

    # Mock model has tensor with name '0:conv2d' and operation 'conv2d' with input port = 1
    # It is setup to return 'is_equal_data=True' for these tensor and port
    # So no collision is expected
    @mock_needed
    def test_decode_name_with_port_delim_equal_data_in(self):
        set_equal_data('conv2d', 'conv2d')
        node = decode_name_with_port(self.model, '0:conv2d')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 1
        assert place_stat.get_input_port == 1
        assert place_stat.is_equal_data > 0
        assert node

    # Stress case: Mock model has:
    # Tensor '8:9'
    # Operation '8:9'
    # Operation '8' with output port = 9
    # Operation '9' with input port = 8
    # All places point to same data - no collision is expected
    @mock_needed
    def test_decode_name_with_port_delim_all_same_data(self):
        set_equal_data('8', '9')
        node = decode_name_with_port(self.model, '8:9')
        model_stat = get_model_statistic()
        place_stat = get_place_statistic()

        assert model_stat.get_place_by_tensor_name == 1
        assert model_stat.get_place_by_operation_name == 2
        assert place_stat.get_input_port == 1
        assert place_stat.get_output_port == 1
        # At least 3 comparisons of places are expected
        assert place_stat.is_equal_data > 2
        assert node
