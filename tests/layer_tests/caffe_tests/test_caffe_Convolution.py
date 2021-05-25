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


def get_conv_params(ie_device=None, precision=None, kernel_size=None, padding=None, stride=None, num_out=None):
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
        kernel_params = [tuple(np.multiply(np.ones(2, int), u)) for u in range(3, 10)]

    if padding:
        padding_params = padding
    else:
        padding_params = [tuple(np.multiply(np.ones(2, int), u)) for u in range(1, 4)]

    if stride:
        stride_params = stride
    else:
        stride_params = [tuple(np.multiply(np.ones(2, int), u)) for u in range(1, 4)]

    if num_out:
        num_out_params = num_out
    else:
        num_out_params = range(4, 10)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, kernel_params, padding_params, stride_params,
                                     num_out_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    scope_for_marker = {
        "precommit": dict(
            kernel_size=[(1, 1), (1, 3)],
            padding=[(1, 1), (1, 3)],
            stride=[(1, 1), (1, 3)],
            num_out=[4, 7]
        )}
    generate_tests(metafunc, get_conv_params, **scope_for_marker)


class TestConvolution(object):
    """
    Temporary disable a lot of tests to speed up caffe suite.
    @generate(*get_conv_params(kernel_size=[(4, 4), (7, 7)]))
    def test_symmetric_kernel(self, ie_device, kernel, pad, stride, nout):
        self.conv(ie_device, kernel, pad, stride, nout)

    @generate(*get_conv_params(kernel_size=[(3, 4), (7, 8)]))
    def test_asymmetric_kernel(self, ie_device, kernel, pad, stride, nout):
        self.conv(ie_device, kernel, pad, stride, nout)

    @generate(*get_conv_params(padding=[(1, 1), (2, 2)]))
    def test_symmetric_padding(self, ie_device, kernel, pad, stride, nout):
        self.conv(ie_device, kernel, pad, stride, nout)

    @generate(*get_conv_params(padding=[(1, 2), (2, 1)]))
    def test_asymmetric_padding(self, ie_device, kernel, pad, stride, nout):
        self.conv(ie_device, kernel, pad, stride, nout)

    @generate(*get_conv_params(stride=[(1, 1), (2, 2)]))
    def test_symmetric_stride(self, ie_device, kernel, pad, stride, nout):
        self.conv(ie_device, kernel, pad, stride, nout)

    @generate(*get_conv_params(stride=[(1, 2), (2, 1)]))
    def test_asymmetric_stride(self, ie_device, kernel, pad, stride, nout):
        self.conv(ie_device, kernel, pad, stride, nout)
    """

    @pytest.mark.precommit
    def test_conv_precommit(self, ie_device, precision, kernel_size, padding, stride, num_out):
        self.conv(ie_device, precision, kernel_size, padding, stride, num_out)

    @pytest.mark.nightly
    def test_conv_nightly(self, ie_device, precision, kernel_size, padding, stride, num_out):
        self.conv(ie_device, precision, kernel_size, padding, stride, num_out)

    def conv(self, ie_device, precision, kernel_size, padding, stride, num_out):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        conv = network.add_layer(layer_type='Convolution',
                                 inputs=[output],
                                 dilations=(1, 1),  # default value
                                 group=1,  # default value
                                 output=num_out,
                                 kernel=kernel_size,
                                 strides=stride,
                                 pads_begin=padding,
                                 pads_end=padding,
                                 get_out_shape_def=caffe_calc_out_shape_conv_layer,
                                 framework_representation_def=conv_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=conv.name)

        assert compare_infer_results_with_caffe(ie_results, conv.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
