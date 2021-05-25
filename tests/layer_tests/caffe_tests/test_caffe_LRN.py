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


def get_lrn_params(ie_device=None, precision=None, alpha=None, beta=None, size=None, region=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param alpha: list if alpha values
    :param beta: list if beta values
    :param size : list if local size values. Should be odd
    :param region : list if region values
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
        alpha_params = [9.99e-06, 1.0, 2]

    if beta:
        beta_params = beta
    else:
        beta_params = [0.75, 1.0, 3]

    if size:
        size_params = size
    else:
        size_params = range(1, 6, 2)

    if region:
        region_params = region
    else:
        region_params = ['across', 'same']

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, alpha_params, beta_params, size_params,
                                     region_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_lrn_params)


class TestLRN(object):
    @pytest.mark.precommit
    def test_lrn_precommit(self, ie_device, precision, alpha, beta, size, region):
        self.lrn(ie_device, precision, alpha, beta, size, region)

    @pytest.mark.nightly
    def test_lrn_nightly(self, ie_device, precision, alpha, beta, size, region):
        self.lrn(ie_device, precision, alpha, beta, size, region)

    def lrn(self, ie_device, precision, alpha, beta, size, region):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        norm = network.add_layer(layer_type='Norm',
                                 inputs=[output],
                                 alpha=alpha,
                                 beta=beta,
                                 local_size=size,
                                 region=region,
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=lrn_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=norm.name)
        assert compare_infer_results_with_caffe(ie_results, norm.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
