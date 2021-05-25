import logging as lg

import pytest
from common.call_InferenceEngine import score_model, compare_infer_results_with_mxnet
from common.call_ModelOptimizer import generate_ir_from_mxnet
from common.constants import *
from common.infer_shapes import *
from common.mxnet_layers_representation import *
from common.legacy.generic_ir_comparator import *
from mxnet_legacy_tests.conftest import generate_tests
from mxnet_legacy_tests.test_mxnet_Convolution import get_conv_params


def pytest_generate_tests(metafunc):
    scope_for_marker = {
        "precommit": dict(
            kernel=[(1, 1), (1, 3)],
            pad=[(1, 1), (1, 3)],
            stride=[(1, 1), (1, 3)],
            num_filter=[1, 5],
            no_bias=[False, True]
        )}
    generate_tests(metafunc, get_conv_params, **scope_for_marker)


class TestDeconvolution(object):
    @pytest.mark.precommit
    def test_deconv_precommit(self, ie_device, precision, kernel, pad, stride, num_filter, no_bias):
        self.deconv(ie_device, precision, kernel, pad, stride, num_filter, no_bias)

    @pytest.mark.nightly
    def test_deconv_nightly(self, ie_device, precision, kernel, pad, stride, num_filter, no_bias):
        self.deconv(ie_device, precision, kernel, pad, stride, num_filter, no_bias)

    def deconv(self, ie_device, precision, kernel, pad, stride, num_filter, no_bias):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        deconv = network.add_layer(layer_type='Deconvolution',
                                   inputs=[inputl],
                                   kernel=kernel,
                                   strides=stride,
                                   pads_begin=pad,
                                   pads_end=pad,
                                   dilations=(1, 1),  # default value
                                   group=1,  # default value
                                   output=num_filter,
                                   no_bias=no_bias,  # ignored parameter
                                   get_out_shape_def=mxnet_calc_out_shape_deconv_layer,
                                   framework_representation_def=deconv_to_symbol)

        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur,
                               ignore_attributes={'Deconvolution': ['no_bias']}), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=deconv.name)

        assert compare_infer_results_with_mxnet(ie_results, network.name, deconv.name,
                                                input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
