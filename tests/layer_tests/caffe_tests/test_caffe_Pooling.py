import itertools
import logging as lg

import pytest
from caffe_tests.conftest import generate_tests
from common.caffe_layers_representation import *
from common.call_InferenceEngine import score_model, compare_infer_results_with_caffe
from common.call_ModelOptimizer import generate_ir_from_caffe
from common.constants import *
from common.infer_shapes import *
from common.legacy.generic_ir_comparator import *


def get_pool_params(ie_device=None, precision=None, kernel_size=None, padding=None, stride=None, method=None):
    """
    :param method: list of pool method values
    :param precision: list of precisions
    :param ie_device: list of devices
    :param kernel_size: list of kernel size values
    :param padding: list of padding values
    :param stride: list of stride values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if kernel_size:
        kernel_params = kernel_size
    else:
        # kernel_size must be > padding else caffe failed with segfault
        kernel_params = [tuple(np.multiply((1, 1), u)) for u in range(3, 10)]

    if padding:
        padding_params = padding
    else:
        padding_params = [tuple(np.multiply((1, 1), u)) for u in range(1, 3)]

    if stride:
        stride_params = stride
    else:
        stride_params = [tuple(np.multiply((1, 1), u)) for u in range(1, 4)]

    if method:
        method_params = method
    else:
        method_params = ['max', 'avg']

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, kernel_params, padding_params, stride_params,
                                     method_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    # kernel_size must be > padding else caffe failed with segfault
    scope_for_marker = {
        "precommit": dict(
            kernel_size=[(3, 3), (3, 5)],
            padding=[(1, 1), (1, 2)],
            stride=[(1, 1), (1, 2)],
        )}
    generate_tests(metafunc, get_pool_params, **scope_for_marker)


class TestPooling(object):
    """
    Temporary disable a lot of tests to speed up caffe suite.
    @unittest.skip('FIXME Cannot convert Pooling layer: *-8879')
    @generate(*get_pool_params(kernel_size=[(3, 4), (4, 3)]))
    def test_asymmetric_kernel(self, ie_device, kernel, pad, stride):
        self.pooling(ie_device, kernel, pad, stride)

    @unittest.skip('FIXME Cannot convert Pooling layer: *-8879')
    @generate(*get_pool_params(kernel_size=[(3, 3), (7, 7)]))
    def test_symmetric_kernel(self, ie_device, kernel, pad, stride):
        self.pooling(ie_device, kernel, pad, stride)

    @unittest.skip('FIXME Cannot convert Pooling layer: *-8879')
    @generate(*get_pool_params(padding=[(1, 1), (2, 2)]))
    def test_symmetric_padding(self, ie_device, kernel, pad, stride):
        self.pooling(ie_device, kernel, pad, stride)

    @unittest.skip('FIXME Cannot convert Pooling layer: *-8879')
    @generate(*get_pool_params(padding=[(2, 1), (1, 2)]))
    def test_asymmetric_padding(self, ie_device, kernel, pad, stride):
        self.pooling(ie_device, kernel, pad, stride)

    @unittest.skip('FIXME Cannot convert Pooling layer: *-8879')
    @generate(*get_pool_params(stride=[(1, 1), (2, 2)]))
    def test_symmetric_stride(self, ie_device, kernel, pad, stride):
        self.pooling(ie_device, kernel, pad, stride)

    @unittest.skip('FIXME Cannot convert Pooling layer: *-8879')
    @generate(*get_pool_params(stride=[(1, 2), (2, 1)]))
    def test_asymmetric_stride(self, ie_device, kernel, pad, stride):
        self.pooling(ie_device, kernel, pad, stride)
    """

    @pytest.mark.precommit
    def test_pooling_precommit(self, ie_device, precision, kernel_size, padding, stride, method):
        self.pooling(ie_device, precision, kernel_size, padding, stride, method)

    @pytest.mark.nightly
    def test_pooling_nightly(self, ie_device, precision, kernel_size, padding, stride, method):
        self.pooling(ie_device, precision, kernel_size, padding, stride, method)

    def pooling(self, ie_device, precision, kernel_size, padding, stride, method):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        pool = network.add_layer(layer_type='Pooling',
                                 inputs=[output],
                                 pool_method=method,
                                 exclude_pad="false",  # default value
                                 rounding_type="ceil",  # default value
                                 kernel=kernel_size,
                                 strides=stride,
                                 pads_begin=padding,
                                 pads_end=padding,
                                 get_out_shape_def=caffe_calc_out_shape_pool_layer,
                                 framework_representation_def=pool_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=pool.name)

        assert compare_infer_results_with_caffe(ie_results, pool.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
