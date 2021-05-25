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


def get_flatten_params(ie_device=None, precision=None, axis=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param axis:  list of pair: axis and end_axis values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if axis:
        axis_params = axis
    else:
        # axis_params = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        axis_params = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, axis_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        if element[0] == 'GPU' and element[2] in [(0, 1), (1, 2), (2, 3)]:
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_flatten_params)


class TestFlatten(object):
    @pytest.mark.precommit
    def test_flatten_precommit(self, ie_device, precision, axis):
        self.flatten(ie_device, precision, axis)

    @pytest.mark.nightly
    def test_flatten_nightly(self, ie_device, precision, axis):
        self.flatten(ie_device, precision, axis)

    def flatten(self, ie_device, precision, axis):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        flatten = network.add_layer(layer_type='Flatten',
                                    inputs=[output],
                                    axis=axis[0],
                                    end_axis=axis[1],
                                    get_out_shape_def=caffe_calc_out_shape_flatten_layer,
                                    framework_representation_def=flatten_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=flatten.name)

        assert compare_infer_results_with_caffe(ie_results, flatten.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
