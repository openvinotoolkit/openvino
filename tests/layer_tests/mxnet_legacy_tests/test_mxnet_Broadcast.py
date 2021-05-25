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


def get_device_params(ie_device=None, precision=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    test_args = []
    for element in itertools.product(ie_device_params, precision_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_device_params)


class TestBroadcast(object):
    @pytest.mark.precommit
    def test_broadcast_mul_precommit(self, ie_device, precision):
        self.broadcast_mul(ie_device, precision)

    @pytest.mark.nightly
    def test_broadcast_mul_nightly(self, ie_device, precision):
        self.broadcast_mul(ie_device, precision)

    def broadcast_mul(self, ie_device, precision):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        weights = network.add_layer(layer_type='Weights',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=weights_to_symbol)
        bcast_mul = network.add_layer(layer_type='ScaleShift',
                                      inputs=[inputl, weights],
                                      get_out_shape_def=calc_same_out_shape,
                                      framework_representation_def=broadcast_mul_to_symbol)
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
                                 image_path=img_path, out_blob_name=bcast_mul.name)

        assert compare_infer_results_with_mxnet(ie_results, network.name, bcast_mul.name,
                                                input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
