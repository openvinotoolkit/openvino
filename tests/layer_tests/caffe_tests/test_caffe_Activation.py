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


def get_power_params(ie_device=None, precision=None, power=None, scale=None, shift=None):
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if power:
        power_params = power
    else:
        power_params = range(-2, 2)

    if scale:
        scale_params = scale
    else:
        scale_params = range(-2, 2)

    if shift:
        shift_params = shift
    else:
        shift_params = range(4, 10)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, power_params, scale_params, shift_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        if element[0] == 'GPU' and element[2] != 1:
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_power_params)


def get_activation_params(ie_device=None, precision=None, activ_type=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param activ_type: list of activation type value
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if activ_type:
        type_params = activ_type
    else:
        type_params = ['sigmoid', 'tanh']

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, type_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "power" in test_func:
        generate_tests(metafunc, get_power_params)
    elif "activation" in test_func:
        generate_tests(metafunc, get_activation_params)


class TestActivation(object):
    @pytest.mark.precommit
    def test_power_precommit(self, ie_device, precision, power, scale, shift):
        self.power(ie_device, precision, power, scale, shift)

    @pytest.mark.nightly
    def test_power_nightly(self, ie_device, precision, power, scale, shift):
        self.power(ie_device, precision, power, scale, shift)

    def power(self, ie_device, precision, power, scale, shift):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        power = network.add_layer(layer_type='Power',
                                  inputs=[output],
                                  power=power,
                                  scale=scale,
                                  shift=shift,
                                  get_out_shape_def=calc_same_out_shape,
                                  framework_representation_def=power_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=power.name)
        assert compare_infer_results_with_caffe(ie_results, power.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_activation_precommit(self, ie_device, precision, activ_type):
        self.activation(ie_device, precision, activ_type)

    @pytest.mark.nightly
    def test_activation_nightly(self, ie_device, precision, activ_type):
        self.activation(ie_device, precision, activ_type)

    def activation(self, ie_device, precision, activ_type):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        activation = network.add_layer(layer_type='Activation',
                                       inputs=[output],
                                       type=activ_type,
                                       get_out_shape_def=calc_same_out_shape,
                                       framework_representation_def=activation_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=activation.name)
        assert compare_infer_results_with_caffe(ie_results, activation.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
