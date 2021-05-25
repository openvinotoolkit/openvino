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


def get_batch_norm_params(ie_device=None, precision=None, epsilon=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param epsilon: list if epsilon values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if epsilon:
        epsilon_params = epsilon
    else:
        epsilon_params = [1e-03, 2e-04, 3e-05, 4e-06, 9.99e-06]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, epsilon_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_batch_norm_params)


def get_network(precision, epsilon):
    network = Net(precision=precision)
    output = network.add_layer(layer_type='Input',
                               get_out_shape_def=calc_out_shape_input_layer,
                               framework_representation_def=input_to_proto)
    network.add_layer(layer_type='BatchNormalization',
                      inputs=[output],
                      epsilon=epsilon,
                      get_out_shape_def=calc_same_out_shape,
                      framework_representation_def=batch_norm_to_proto)
    return network


class TestBatchNormalization(object):
    @pytest.mark.precommit
    def test_batch_normalization_precommit(self, ie_device, precision, epsilon):
        self.batch_normalization(ie_device, precision, epsilon)

    @pytest.mark.nightly
    def test_batch_normalization_nightly(self, ie_device, precision, epsilon):
        self.batch_normalization(ie_device, precision, epsilon)

    def batch_normalization(self, ie_device, precision, epsilon):
        network = get_network(precision, epsilon)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='BatchNormalization1')
        assert compare_infer_results_with_caffe(ie_results, 'BatchNormalization1'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_batch_normalization_fused_precommit(self, ie_device, precision, epsilon):
        self.batch_normalization(ie_device, precision, epsilon)

    @pytest.mark.nightly
    def test_batch_normalization_fused_nightly(self, ie_device, precision, epsilon):
        self.batch_normalization(ie_device, precision, epsilon)

    def batch_normalization_fused(self, ie_device, precision, epsilon):
        network = get_network(precision, epsilon)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='Mul_10-FusedScaleShift_13')
        assert compare_infer_results_with_caffe(ie_results, 'BatchNormalization1'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
