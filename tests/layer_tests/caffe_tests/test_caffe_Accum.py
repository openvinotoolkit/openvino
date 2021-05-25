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


def get_accum_params(ie_device=None, precision=None, top=None, size_divisible_by=None, have_reference=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param top: list of set of top value [(...)]
    :param size_divisible_by: list of set of size_divisible_by value [(...)]
    :param have_reference: list of set of have_reference value [(...)]
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if top:
        top_params = top
    else:
        top_params = [(222, 222), (224, 224), (226, 226)]

    if size_divisible_by:
        size_divisible_by_params = size_divisible_by
    else:
        size_divisible_by_params = [0, 1]

    if have_reference:
        have_reference_params = have_reference
    else:
        have_reference_params = [1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, top_params, size_divisible_by_params,
                                     have_reference_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_accum_params)


class TestAccum(object):
    @pytest.mark.precommit
    def test_accum_precommit(self, ie_device, precision, top, size_divisible_by, have_reference):
        self.accum(ie_device, precision, top, size_divisible_by, have_reference)

    @pytest.mark.nightly
    def test_accum_nightly(self, ie_device, precision, top, size_divisible_by, have_reference):
        self.accum(ie_device, precision, top, size_divisible_by, have_reference)

    def accum(self, ie_device, precision, top, size_divisible_by, have_reference):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        reference = network.add_layer(layer_type='Input',
                                      get_out_shape_def=calc_out_shape_input_layer,
                                      framework_representation_def=input_to_proto)
        accum = network.add_layer(layer_type='Accum',
                                  inputs=[output, reference],
                                  top_width=top[0],
                                  top_height=top[1],
                                  size_divisible_by=size_divisible_by,
                                  have_reference=have_reference,
                                  get_out_shape_def=caffe_calc_out_accum_layer,
                                  framework_representation_def=accum_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=accum.name)
        assert compare_infer_results_with_caffe(ie_results, accum.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
