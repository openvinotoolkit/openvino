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


def get_scale_shift_params(ie_device=None, precision=None):
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


def get_network(precision):
    network = Net(precision=precision)
    output = network.add_layer(layer_type='Input',
                               get_out_shape_def=calc_out_shape_input_layer,
                               framework_representation_def=input_to_proto)
    network.add_layer(layer_type='ScaleShift',
                      inputs=[output],
                      get_out_shape_def=calc_same_out_shape,
                      framework_representation_def=scale_to_proto)
    return network


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_scale_shift_params)


class TestScaleShift(object):
    @pytest.mark.precommit
    def test_scale_shift_fused_precommit(self, ie_device, precision):
        self.scale_shift_fused(ie_device, precision)

    @pytest.mark.nightly
    def test_scale_shift_fused_nightly(self, ie_device, precision):
        self.scale_shift_fused(ie_device, precision)

    def scale_shift_fused(self, ie_device, precision):
        network = get_network(precision)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='ScaleShift1-Mul_10')

        assert compare_infer_results_with_caffe(ie_results, 'ScaleShift1'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_scale_shift_precommit(self, ie_device, precision):
        self.scale_shift(ie_device, precision)

    @pytest.mark.nightly
    def test_scale_shift_nightly(self, ie_device, precision):
        self.scale_shift(ie_device, precision)

    def scale_shift(self, ie_device, precision):
        network = get_network(precision)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='ScaleShift1')

        assert compare_infer_results_with_caffe(ie_results, 'ScaleShift1'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
