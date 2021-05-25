import itertools
import logging as lg

import pytest
from common.call_InferenceEngine import score_model, compare_infer_results_with_mxnet
from common.call_ModelOptimizer import generate_ir_from_mxnet
from common.constants import *
from common.infer_shapes import *
from common.mxnet_layers_representation import *
from common.legacy.generic_ir_comparator import *
from mxnet_legacy_tests.conftest import generate_tests


def get_conv_params(ie_device=None, precision=None, kernel=None, pad=None, stride=None, num_filter=None, no_bias=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param kernel: list of tuples of kernel values
    :param pad: list of tuples of padding values
    :param stride: list of tuples of stride values
    :param num_filter: list of num_filter values
    :param no_bias: list of bool values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if kernel:
        kernel_params = kernel
    else:
        kernel_params = [tuple(np.multiply((1, 1), u)) for u in range(3, 10)]

    if pad:
        padding_params = pad
    else:
        padding_params = [tuple(np.multiply(np.ones(2, int), u)) for u in range(1, 4)]

    if stride:
        stride_params = stride
    else:
        stride_params = [tuple(np.multiply(np.ones(2, int), u)) for u in range(1, 4)]

    if num_filter:
        num_filter_params = num_filter
    else:
        num_filter_params = range(4, 10)

    if no_bias:
        no_bias_params = no_bias
    else:
        no_bias_params = [False, True]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, kernel_params,
                                     padding_params, stride_params, num_filter_params, no_bias_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    scope_for_marker = {
        "precommit": dict(
            kernel=[(1, 1), (1, 3)],
            pad=[(1, 1), (1, 3)],
            stride=[(1, 1), (1, 3)],
            num_filter=[4, 7],
            no_bias=[False, True]
        )}
    generate_tests(metafunc, get_conv_params, **scope_for_marker)


class TestConvolution(object):
    @pytest.mark.precommit
    def test_conv_precommit(self, ie_device, precision, kernel, pad, stride, num_filter, no_bias):
        self.conv(ie_device, precision, kernel, pad, stride, num_filter, no_bias)

    @pytest.mark.nightly
    def test_conv_nightly(self, ie_device, precision, kernel, pad, stride, num_filter, no_bias):
        self.conv(ie_device, precision, kernel, pad, stride, num_filter, no_bias)

    def conv(self, ie_device, precision, kernel, pad, stride, num_filter, no_bias):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        conv = network.add_layer(layer_type='Convolution',
                                 inputs=[inputl],
                                 kernel=kernel,
                                 strides=stride,
                                 pads_begin=pad,
                                 pads_end=pad,
                                 dilations=(1, 1),  # default value
                                 group=1,  # default value
                                 output=num_filter,
                                 no_bias=no_bias,  # ignored parameter
                                 get_out_shape_def=mxnet_calc_out_shape_conv_layer,
                                 framework_representation_def=conv_to_symbol)

        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur,
                               ignore_attributes={'Convolution': ['no_bias']}), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=conv.name)

        assert compare_infer_results_with_mxnet(ie_results, network.name, conv.name,
                                                input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
