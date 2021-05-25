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


def get_prior_box_params(ie_device=None, precision=None, step=None, min_size=None, max_size=None, flip=None, clip=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param step: list of step values
    :param min_size: list of min values
    :param max_size: list of max values
    :param flip: list of flip values
    :param clip: list of clip values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if step:
        step_params = step
    else:
        step_params = range(1, 101, 37)

    if min_size:
        min_params = min_size
    else:
        min_params = range(1, 162, 54)

    if max_size:
        max_params = max_size
    else:
        max_params = range(162, 224, 18)

    if flip:
        flip_params = flip
    else:
        flip_params = [0, 1]

    if clip:
        clip_params = clip
    else:
        clip_params = [0, 1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, step_params, min_params, max_params,
                                     flip_params, clip_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def get_prior_box_clustered_params(ie_device=None, precision=None, size=None, img=None, step=None, offset=None,
                                   clip=None, flip=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param size: list of size values
    :param img: list of img values
    :param step: list of step values
    :param offset: list of offset values
    :param clip: list of clip values
    :param flip: list of flip values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if size:
        size_params = size
    else:
        size_params = [(222, 222), (224, 224), (226, 226)]

    if img:
        img_params = img
    else:
        img_params = size_params

    if step:
        step_params = step
    else:
        step_params = [(1, 1), (2, 2), (10, 10)]

    if offset:
        offset_params = offset
    else:
        offset_params = [0, 1]

    if clip:
        clip_params = clip
    else:
        clip_params = [0, 1]

    if flip:
        flip_params = flip
    else:
        flip_params = [0, 1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, size_params, img_params, step_params,
                                     offset_params,
                                     clip_params, flip_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "prior_box" in test_func and "clustered" not in test_func:
        scope_for_marker = {
            "precommit": dict(
                step=range(1, 101, 74),
                min_size=range(1, 162, 108),
                max_size=range(162, 224, 30)
            )}
        generate_tests(metafunc, get_prior_box_params, **scope_for_marker)
    elif "prior_box_clustered" in test_func:
        scope_for_marker = {
            "precommit": dict(
                size=[(222, 222)],
                step=[(1, 1), (2, 2)]
            )}
        generate_tests(metafunc, get_prior_box_clustered_params, **scope_for_marker)


class TestPriorBox(object):
    @pytest.mark.precommit
    def test_prior_box_precommit(self, ie_device, precision, step, min_size, max_size, flip, clip):
        self.prior_box(ie_device, precision, step, min_size, max_size, flip, clip)

    @pytest.mark.nightly
    def test_prior_box_nightly(self, ie_device, precision, step, min_size, max_size, flip, clip):
        self.prior_box(ie_device, precision, step, min_size, max_size, flip, clip)

    def prior_box(self, ie_device, precision, step, min_size, max_size, flip, clip):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[output],
                                 negative_slope=1,
                                 engine='caffe.ReLUParameter.DEFAULT',
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_proto)
        priorbox = network.add_layer(layer_type='PriorBox',
                                     inputs=[output, relu],
                                     step=step,
                                     min_size=min_size,
                                     max_size=max_size,
                                     offset=0.5,
                                     aspect_ratio='2.0,3.0',
                                     flip=flip,
                                     clip=clip,
                                     variance="0.10000000149011612,0.10000000149011612,"
                                              "0.20000000298023224,0.20000000298023224",
                                     get_out_shape_def=caffe_calc_out_shape_prior_box_layer,
                                     framework_representation_def=prior_box_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=priorbox.name)

        assert compare_infer_results_with_caffe(ie_results, priorbox.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_prior_box_clustered_precommit(self, ie_device, precision, size, img, step, offset, clip, flip):
        self.prior_box_clustered(ie_device, precision, size, img, step, offset, clip, flip)

    @pytest.mark.nightly
    def test_prior_box_clustered_nightly(self, ie_device, precision, size, img, step, offset, clip, flip):
        self.prior_box_clustered(ie_device, precision, size, img, step, offset, clip, flip)

    def prior_box_clustered(self, ie_device, precision, size, img, step, offset, clip, flip):
        network = Net(precision=test_precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[output],
                                 negative_slope=1,
                                 engine='caffe.ReLUParameter.DEFAULT',
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_proto)
        priorbox_cl = network.add_layer(layer_type='PriorBoxClustered',
                                        inputs=[output, relu],
                                        width=size[0],
                                        height=size[1],
                                        img_w=img[0],
                                        img_h=img[1],
                                        step_w=step[0],
                                        step_h=step[1],
                                        offset=offset,
                                        clip=clip,
                                        flip=flip,
                                        variance="0.10000000149011612,0.10000000149011612,"
                                                 "0.20000000298023224,0.20000000298023224",
                                        get_out_shape_def=caffe_calc_out_shape_prior_box_clustered_layer,
                                        framework_representation_def=prior_box_clustered_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=priorbox_cl.name)

        assert compare_infer_results_with_caffe(ie_results, priorbox_cl.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
