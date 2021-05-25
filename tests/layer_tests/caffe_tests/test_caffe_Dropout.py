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


def get_dropout_params(ie_device=None, precision=None, ratio=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param ratio: list of set of ratio value [(...)]
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if ratio:
        ratio_params = ratio
    else:
        ratio_params = [0, 0.1, 0.25, 0.5, 0.75, 1]

    test_args = []

    for element in itertools.product(ie_device_params, precision_params, ratio_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_dropout_params)


class TestDropout(object):
    @pytest.mark.precommit
    def test_dropout_precommit(self, ie_device, precision, ratio):
        self.dropout(ie_device, precision, ratio)

    @pytest.mark.nightly
    def test_dropout_nightly(self, ie_device, precision, ratio):
        self.dropout(ie_device, precision, ratio)

    def dropout(self, ie_device, precision, ratio):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        dropout = network.add_layer(layer_type='Dropout',
                                    inputs=[output],
                                    dropout_ratio=ratio,
                                    get_out_shape_def=calc_same_out_shape,
                                    framework_representation_def=dropout_to_proto)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[dropout],
                                 negative_slope=0,
                                 engine='caffe.ReLUParameter.DEFAULT',
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=relu.name)

        assert compare_infer_results_with_caffe(ie_results, relu.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
