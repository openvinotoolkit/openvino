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


def get_mvn_params(ie_device=None, precision=None, normalize_variance=None, across_channels=None, eps=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param normalize_variance: list if normalize_variance values
        This parameter can be set to false to normalize mean only
    :param across_channels: list if across_channels values
        This parameter can be set to true to perform DNN-like MVN
    :param eps : list if eps values.
        Epsilon for not dividing by zero while normalizing variance
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if normalize_variance:
        normalize_params = normalize_variance
    else:
        normalize_params = [0, 1]

    if across_channels:
        channels_params = across_channels
    else:
        channels_params = [0, 1]

    if eps:
        eps_params = eps
    else:
        eps_params = [1e-03, 9.99e-06]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, normalize_params, channels_params, eps_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_mvn_params)


class TestMVN(object):
    @pytest.mark.precommit
    def test_mvn_precommit(self, ie_device, precision, normalize_variance, across_channels, eps):
        self.mvn(ie_device, precision, normalize_variance, across_channels, eps)

    @pytest.mark.nightly
    def test_mvn_nightly(self, ie_device, precision, normalize_variance, across_channels, eps):
        self.mvn(ie_device, precision, normalize_variance, across_channels, eps)

    def mvn(self, ie_device, precision, normalize_variance, across_channels, eps):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        mvn = network.add_layer(layer_type='MVN',
                                inputs=[output],
                                normalize_variance=normalize_variance,
                                across_channels=across_channels,
                                eps=eps,
                                get_out_shape_def=calc_same_out_shape,
                                framework_representation_def=mvn_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=mvn.name)
        assert compare_infer_results_with_caffe(ie_results, mvn.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
