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


def get_transpose_params(ie_device=None, precision=None, order=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param order: list of order
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if order:
        order_params = order
    else:
        order_params = itertools.permutations([0, 1, 2, 3])

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, order_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        if element[0] == 'MYRIAD':
            if element[1] == 'FP32':
                continue
        if element[2] == (0, 1, 2, 3):
            # Skip this case due *_12805
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_transpose_params)


class TestPermute(object):
    @pytest.mark.precommit
    def test_permute_precommit(self, ie_device, precision, order):
        self.permute(ie_device, precision, order)

    @pytest.mark.nightly
    def test_permute_nightly(self, ie_device, precision, order):
        self.permute(ie_device, precision, order)

    def permute(self, ie_device, precision, order):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        transpose = network.add_layer(layer_type='Permute',
                                      inputs=[inputl],
                                      order=order,
                                      get_out_shape_def=caffe_calc_out_shape_permute_layer,
                                      framework_representation_def=transpose_to_symbol)
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

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=transpose.name)

        assert compare_infer_results_with_mxnet(ie_results, network.name, transpose.name,
                                                input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
