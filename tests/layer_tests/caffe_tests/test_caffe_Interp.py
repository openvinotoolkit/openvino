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


def get_interp_base_params(ie_device=None, precision=None, pad_beg=None, pad_end=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param pad_beg: list if pad_beg values
    :param pad_end: list if pad_end values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if pad_beg:
        pad_beg_params = pad_beg
    else:
        pad_beg_params = range(-4, 1)

    if pad_end:
        pad_end_params = pad_end
    else:
        pad_end_params = range(-4, 1)

    return ie_device_params, precision_params, pad_beg_params, pad_end_params


def get_interp_params():
    ie_device_params, precision_params, pad_beg_params, pad_end_params = get_interp_base_params()

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, pad_beg_params, pad_end_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def get_interp_params_with_shrink(ie_device=None, precision=None, pad_beg=None, pad_end=None, shrink=None):
    ie_device_params, precision_params, pad_beg_params, pad_end_params = get_interp_base_params(ie_device, precision,
                                                                                                pad_beg, pad_end)

    if shrink:
        shrink_param = shrink
    else:
        shrink_param = range(2, 4)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, pad_beg_params, pad_end_params, shrink_param):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def get_interp_params_with_zoom(ie_device=None, precision=None, pad_beg=None, pad_end=None, zoom=None):
    ie_device_params, precision_params, pad_beg_params, pad_end_params = get_interp_base_params(ie_device, precision,
                                                                                                pad_beg, pad_end)

    if zoom:
        zoom_param = zoom
    else:
        zoom_param = range(2, 4)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, pad_beg_params, pad_end_params, zoom_param):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def get_interp_params_with_width_height(ie_device=None, precision=None, pad_beg=None, pad_end=None, width_height=None):
    ie_device_params, precision_params, pad_beg_params, pad_end_params = get_interp_base_params(ie_device, precision,
                                                                                                pad_beg, pad_end)

    if width_height:
        width_height_params = width_height
    else:
        width_height_params = [tuple(np.multiply(np.ones(2, int), u)) for u in [200, 222, 224, 226, 228]]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, pad_beg_params, pad_end_params,
                                     width_height_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    scope_for_marker = {
        "precommit": dict(
            pad_beg=range(-1, 1),
            pad_end=range(-1, 1)
        )}

    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "interp_with_shrink" in test_func:
        generate_tests(metafunc, get_interp_params_with_shrink, **scope_for_marker)
    elif "interp_with_zoom" in test_func:
        generate_tests(metafunc, get_interp_params_with_zoom, **scope_for_marker)
    elif "interp_with_width_height_only" in test_func:
        generate_tests(metafunc, get_interp_params_with_width_height, **scope_for_marker)


class TestInterp(object):
    @pytest.mark.precommit
    def test_interp_with_shrink_precommit(self, ie_device, precision, pad_beg, pad_end, shrink):
        self.interp(ie_device, precision, pad_beg, pad_end, shrink, 1, (0, 0))

    @pytest.mark.nightly
    def test_interp_with_shrink_nightly(self, ie_device, precision, pad_beg, pad_end, shrink):
        self.interp(ie_device, precision, pad_beg, pad_end, shrink, 1, (0, 0))

    @pytest.mark.precommit
    def test_interp_with_zoom_precommit(self, ie_device, precision, pad_beg, pad_end, zoom):
        self.interp(ie_device, precision, pad_beg, pad_end, 1, zoom, (0, 0))

    @pytest.mark.nightly
    def test_interp_with_zoom_nightly(self, ie_device, precision, pad_beg, pad_end, zoom):
        self.interp(ie_device, precision, pad_beg, pad_end, 1, zoom, (0, 0))

    @pytest.mark.precommit
    def test_interp_with_width_height_only_precommit(self, ie_device, precision, pad_beg, pad_end, width_height):
        self.interp(ie_device, precision, pad_beg, pad_end, 1, 1, width_height)

    @pytest.mark.nightly
    def test_interp_with_width_height_only_nightly(self, ie_device, precision, pad_beg, pad_end, width_height):
        self.interp(ie_device, precision, pad_beg, pad_end, 1, 1, width_height)

    def interp(self, ie_device, precision, pad_beg, pad_end, shrink, zoom, width_height):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        interp = network.add_layer(layer_type='Interp',
                                   inputs=[output],
                                   pad_beg=pad_beg,
                                   pad_end=pad_end,
                                   shrink_factor=shrink,
                                   zoom_factor=zoom,
                                   height=width_height[1],
                                   width=width_height[0],
                                   get_out_shape_def=caffe_calc_out_shape_interp_layer,
                                   framework_representation_def=interp_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=interp.name)

        assert compare_infer_results_with_caffe(ie_results, interp.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
