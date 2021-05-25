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


def get_eltwise_params(ie_device=None, precision=None, operation=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param operation: list of operation types
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if operation:
        operation_params = operation
    else:
        operation_params = ['sum', 'mul']

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, operation_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_eltwise_params)


class TestEltwise(object):
    @pytest.mark.precommit
    def test_eltwise_precommit(self, ie_device, precision, operation):
        self.eltwise(ie_device, precision, operation)

    @pytest.mark.nightly
    def test_eltwise_nightly(self, ie_device, precision, operation):
        self.eltwise(ie_device, precision, operation)

    def eltwise(self, ie_device, precision, operation):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[inputl],
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_symbol)
        eltwise = network.add_layer(layer_type='Eltwise',
                                    inputs=[inputl, relu],
                                    operation=operation,
                                    get_out_shape_def=calc_same_out_shape,
                                    framework_representation_def=eltwise_to_symbol)
        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device, batch=1,
                                 image_path=img_path, out_blob_name=eltwise.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, eltwise.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
