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


def get_device(ie_device=None, precision=None):
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


def get_greedy_ctc_decoder_params(ie_device=None, precision=None, ctc_merge_repeated=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param ctc_merge_repeated: list of set of ctc_merge_related value [(...)]
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if ctc_merge_repeated:
        ctc_merge_repeated_params = ctc_merge_repeated
    else:
        ctc_merge_repeated_params = [0, 1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, ctc_merge_repeated_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "ctc_greedy_decoder" in test_func:
        generate_tests(metafunc, get_greedy_ctc_decoder_params)
    elif "ctc_beam_search" in test_func:
        generate_tests(metafunc, get_device)


class TestCTCDecoder(object):
    @pytest.mark.precommit
    def test_ctc_greedy_decoder_two_bottom_precommit(self, ie_device, precision, ctc_merge_repeated):
        self.ctc_greedy_decoder_two_bottom(ie_device, precision, ctc_merge_repeated)

    @pytest.mark.nightly
    def test_ctc_greedy_decoder_two_bottom_nightly(self, ie_device, precision, ctc_merge_repeated):
        self.ctc_greedy_decoder_two_bottom(ie_device, precision, ctc_merge_repeated)

    def ctc_greedy_decoder_two_bottom(self, ie_device, precision, ctc_merge_repeated):
        network = Net(precision=precision)
        output1 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        output2 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        dec = network.add_layer(layer_type='CTCGreedyDecoder',
                                inputs=[output1, output2],
                                ctc_merge_repeated=ctc_merge_repeated,
                                get_out_shape_def=caffe_calc_out_shape_ctc_decoder_layer,
                                framework_representation_def=ctc_greedy_decoder_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=dec.name)

        assert compare_infer_results_with_caffe(ie_results, dec.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.skip("Model Optimizer does not support this layer")
    @pytest.mark.precommit
    def test_ctc_beam_search_decoder_two_bottom_precommit(self, ie_device, precision):
        self.ctc_beam_search_decoder_two_bottom(ie_device, precision)

    @pytest.mark.skip("Model Optimizer does not support this layer")
    @pytest.mark.nightly
    def test_ctc_beam_search_decoder_two_bottom_nightly(self, ie_device, precision):
        self.ctc_beam_search_decoder_two_bottom(ie_device, precision)

    def ctc_beam_search_decoder_two_bottom(self, ie_device, precision):
        network = Net(precision=precision)
        output1 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        output2 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        network.add_layer(layer_type='CTCBeamSearchDecoder',
                          inputs=[output1, output2],
                          get_out_shape_def=caffe_calc_out_shape_ctc_decoder_layer,
                          framework_representation_def=ctc_beam_search_decoder_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='CTCBeamSearchDecoder2')

        assert compare_infer_results_with_caffe(ie_results, 'CTCBeamSearchDecoder2'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_ctc_greedy_decoder_three_bottom_precommit(self, ie_device, precision, ctc_merge_repeated):
        self.ctc_greedy_decoder_three_bottom(ie_device, precision, ctc_merge_repeated)

    @pytest.mark.nightly
    def test_ctc_greedy_decoder_three_bottom_nightly(self, ie_device, precision, ctc_merge_repeated):
        self.ctc_greedy_decoder_three_bottom(ie_device, precision, ctc_merge_repeated)

    def ctc_greedy_decoder_three_bottom(self, ie_device, precision, ctc_merge_repeated):
        network = Net(precision=precision)
        output1 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        output2 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        output3 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        network.add_layer(layer_type='CTCGreedyDecoder',
                          inputs=[output1, output2, output3],
                          ctc_merge_repeated=ctc_merge_repeated,
                          get_out_shape_def=caffe_calc_out_shape_ctc_decoder_layer,
                          framework_representation_def=ctc_greedy_decoder_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='CTCBeamSearchDecoder3')

        assert compare_infer_results_with_caffe(ie_results, 'CTCBeamSearchDecoder3'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.skip("Model Optimizer does not support this layer")
    @pytest.mark.precommit
    def test_ctc_beam_search_decoder_three_bottom_precommit(self, ie_device, precision):
        self.ctc_beam_search_decoder_three_bottom(ie_device, precision)

    @pytest.mark.skip("Model Optimizer does not support this layer")
    @pytest.mark.nightly
    def test_ctc_beam_search_decoder_three_bottom_nightly(self, ie_device, precision):
        self.ctc_beam_search_decoder_three_bottom(ie_device, precision)

    def ctc_beam_search_decoder_three_bottom(self, ie_device, precision):
        network = Net(precision=precision)
        output1 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        output2 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        output3 = network.add_layer(layer_type='Input',
                                    get_out_shape_def=calc_out_shape_input_layer,
                                    framework_representation_def=input_to_proto)
        network.add_layer(layer_type='CTCBeamSearchDecoder',
                          inputs=[output1, output2, output3],
                          get_out_shape_def=caffe_calc_out_shape_ctc_decoder_layer,
                          framework_representation_def=ctc_beam_search_decoder_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")
