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


def get_l2norm_params(ie_device=None, precision=None, eps=None, mode=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param eps: list of eps values
    :param mode: list of mode values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if eps:
        eps_params = eps
    else:
        eps_params = [9.99e-06, 1e-10, 2]  # 1e-10 is default

    if mode:
        mode_params = mode
    else:
        mode_params = ['channel', 'instance', 'spatial']  # 'instance' is default

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, eps_params, mode_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_l2norm_params)


class TestL2Normalization(object):
    @pytest.mark.precommit
    def test_l2norm_precommit(self, ie_device, precision, eps, mode):
        self.l2norm(ie_device, precision, eps, mode)

    @pytest.mark.nightly
    def test_l2norm_nightly(self, ie_device, precision, eps, mode):
        self.l2norm(ie_device, precision, eps, mode)

    def l2norm(self, ie_device, precision, eps, mode):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        l2norm = network.add_layer(layer_type='l2normalization',
                                   inputs=[inputl],
                                   eps=eps,
                                   mode=mode,
                                   across_spatial=0,  # default value
                                   channel_shared=0,  # default value
                                   get_out_shape_def=calc_same_out_shape,
                                   framework_representation_def=l2normalization_to_symbol)
        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur,
                               ignore_attributes={'l2normalization': ['mode']}), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device, batch=1,
                                 image_path=img_path, out_blob_name=l2norm.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, l2norm.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
