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


def get_arg_max_params(ie_device=None, precision=None, axis=None, out_max=None, top_k=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param axis: list of set of axis value [(...)]
    :param out_max: list of set of out_max value [(...)]
    :param top_k: list of set of top_k value [(...)]
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
        axis_params = range(1, 4)

    if out_max:
        out_max_params = out_max
    else:
        out_max_params = [0, 1]

    if top_k:
        top_k_params = top_k
    else:
        top_k_params = [1, 2]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, axis_params, out_max_params, top_k_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_arg_max_params)


class TestArgMax(object):
    @pytest.mark.precommit
    def test_arg_max_precommit(self, ie_device, precision, axis, out_max, top_k):
        self.arg_max(ie_device, precision, axis, out_max, top_k)

    @pytest.mark.nightly
    def test_arg_max_nightly(self, ie_device, precision, axis, out_max, top_k):
        self.arg_max(ie_device, precision, axis, out_max, top_k)

    def arg_max(self, ie_device, precision, axis, out_max, top_k):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        argmax = network.add_layer(layer_type='ArgMax',
                                   inputs=[output],
                                   axis=axis,
                                   out_max_val=out_max,
                                   top_k=top_k,
                                   get_out_shape_def=caffe_calc_out_arg_max_layer,
                                   framework_representation_def=arg_max_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=argmax.name)

        assert compare_infer_results_with_caffe(ie_results, argmax.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
