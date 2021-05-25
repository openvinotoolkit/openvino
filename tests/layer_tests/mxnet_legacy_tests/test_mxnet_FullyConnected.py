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


def get_fc_params(ie_device=None, precision=None, nout=None, no_bias=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param nout: list of nout values
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

    if nout:
        nout_params = nout
    else:
        nout_params = range(1, 150, 7)

    if no_bias:
        no_bias_params = no_bias
    else:
        no_bias_params = [False, True]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, nout_params, no_bias_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_fc_params)


class TestFullyConnected(object):
    @pytest.mark.precommit
    def test_fullyconnected_precommit(self, ie_device, precision, nout, no_bias):
        self.fullyconnected(ie_device, precision, nout, no_bias)

    @pytest.mark.nightly
    def test_fullyconnected_nightly(self, ie_device, precision, nout, no_bias):
        self.fullyconnected(ie_device, precision, nout, no_bias)

    def fullyconnected(self, ie_device, precision, nout, no_bias):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        fully_conected = network.add_layer(layer_type='FullyConnected',
                                           inputs=[inputl],
                                           out_size=nout,
                                           no_bias=no_bias,  # ignored parameter
                                           get_out_shape_def=caffe_calc_out_shape_fullyconnected_layer,
                                           framework_representation_def=fullyconected_to_symbol)

        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur,
                               ignore_attributes={'FullyConnected': ['no_bias']}), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=fully_conected.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, fully_conected.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
