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


def get_lrn_params(ie_device=None, precision=None, alpha=None, beta=None, knorm=None, size=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param alpha: list of alpha values
    :param beta: list of beta values
    :param knorm: list of knorm values
    :param size : list of local size values. Should be odd
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if alpha:
        alpha_params = alpha
    else:
        alpha_params = [9.99e-06, 0.0001, 2]  # 0.0001 is default

    if beta:
        beta_params = beta
    else:
        beta_params = [0.75, 1.0, 3]  # 0.75 is default

    if knorm:
        knorm_params = knorm
    else:
        knorm_params = [1.0, 2]  # 2 is default

    if size:
        size_params = size
    else:
        # TODO: IE doesn't support even (2, 4...) values
        # size_params = range(1, 6, 2)
        # TODO: If size_params > 3, MXNet fails with error
        size_params = range(1, 4, 2)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, alpha_params, beta_params,
                                     knorm_params, size_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_lrn_params)


class TestLRN(object):
    @pytest.mark.precommit
    def test_lrn_precommit(self, ie_device, precision, alpha, beta, knorm, size):
        self.lrn(ie_device, precision, alpha, beta, knorm, size)

    @pytest.mark.nightly
    def test_lrn_nightly(self, ie_device, precision, alpha, beta, knorm, size):
        self.lrn(ie_device, precision, alpha, beta, knorm, size)

    def lrn(self, ie_device, precision, alpha, beta, knorm, size):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        norm = network.add_layer(layer_type='Norm',
                                 inputs=[inputl],
                                 alpha=alpha,
                                 beta=beta,
                                 knorm=knorm,
                                 local_size=size,
                                 region="across",  # default value
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=lrn_to_symbol)
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
                                 image_path=img_path, out_blob_name=norm.name)

        assert compare_infer_results_with_mxnet(ie_results, network.name, norm.name,
                                                input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
