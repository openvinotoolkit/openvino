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


def get_relu_params(ie_device=None, precision=None, nslope=None, engine=None):
    """
    The ReLU layer computes the output as x if x > 0 and negative_slope * x if x <= 0.
    When the negative slope parameter is not set, it is equivalent to the standard ReLU function of taking max(x, 0).
    It also supports in-place computation,
    meaning that the bottom and the top blob could be the same to preserve memory consumption.
    :param ie_device: list of test devices
    :param precision: list of precisions
    :param nslope: list if nslope values
    :param engine:
        caffe.ReLUParameter.DEFAULT
        caffe.ReLUParameter.CAFFE
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if nslope:
        nslope_params = nslope
    else:
        nslope_params = range(-2, 2)

    if engine:
        engine_params = engine
    else:
        # TODO: Check why 'caffe.ReLUParameter.CUDNN' isn't supported
        engine_params = ['caffe.ReLUParameter.DEFAULT',
                         'caffe.ReLUParameter.CAFFE']

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, nslope_params, engine_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def get_prelu_params(ie_device=None, precision=None, _min=None, _max=None, mean=None, sparse=None, std=None,
                     variance_norm=None, channel_shared=None):
    """
    :param ie_device: list of test devices
    :param precision: list of precisions
    list of filler params:
    :param _min
    :param _max
    :param mean
    :param sparse
    :param std
    :param variance_norm
        caffe.FillerParameter.FAN_IN
        caffe.FillerParameter.FAN_OUT
        caffe.FillerParameter.AVERAGE
    :param channel_shared: list of channel_shared params. 0 or 1
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if _min:
        min_params = _min
    else:
        min_params = range(0, 2)

    if _max:
        max_params = _max
    else:
        max_params = range(0, 2)

    if mean:
        mean_params = mean
    else:
        mean_params = range(0, 2)

    if sparse:
        sparse_params = sparse
    else:
        sparse_params = [-1]

    if std:
        std_params = std
    else:
        std_params = range(0, 2)

    if variance_norm:
        variance_norm_params = variance_norm
    else:
        variance_norm_params = ['caffe.FillerParameter.FAN_IN',
                                'caffe.FillerParameter.FAN_OUT',
                                'caffe.FillerParameter.AVERAGE']

    if channel_shared:
        channel_shared_params = channel_shared
    else:
        channel_shared_params = [0, 1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, min_params, max_params, mean_params,
                                     sparse_params, std_params,
                                     variance_norm_params, channel_shared_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "_relu" in test_func:
        generate_tests(metafunc, get_relu_params)
    elif "prelu" in test_func:
        generate_tests(metafunc, get_prelu_params)


class TestReLU(object):
    @pytest.mark.precommit
    def test_relu_precommit(self, ie_device, precision, nslope, engine):
        self.relu(ie_device, precision, nslope, engine)

    @pytest.mark.nightly
    def test_relu_nightly(self, ie_device, precision, nslope, engine):
        self.relu(ie_device, precision, nslope, engine)

    def relu(self, ie_device, precision, nslope, engine):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[output],
                                 negative_slope=nslope,
                                 engine=engine,
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))
        ignore = {'ReLU': ['engine']} if nslope != 0 else {'ReLU': ['engine', 'negative_slope']}

        assert network.compare(network_cur, ignore_attributes=ignore), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=relu.name)

        assert compare_infer_results_with_caffe(ie_results, relu.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_prelu_precommit(self, ie_device, precision, _min, _max, mean, sparse, std, variance_norm, channel_shared):
        self.prelu(ie_device, precision, _min, _max, mean, sparse, std, variance_norm, channel_shared)

    @pytest.mark.nightly
    def test_prelu_nightly(self, ie_device, precision, _min, _max, mean, sparse, std, variance_norm, channel_shared):
        self.prelu(ie_device, precision, _min, _max, mean, sparse, std, variance_norm, channel_shared)

    def prelu(self, ie_device, precision, _min, _max, mean, sparse, std, variance_norm, channel_shared):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        prelu = network.add_layer(layer_type='PReLU',
                                  inputs=[output],
                                  min=_min,
                                  max=_max,
                                  mean=mean,
                                  sparse=sparse,
                                  std=std,
                                  variance_norm=variance_norm,
                                  channel_shared=channel_shared,
                                  get_out_shape_def=calc_same_out_shape,
                                  framework_representation_def=prelu_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes={
            'PReLU': ['channel_shared']}), "Comparing of networks failed."  # FIXME: Delete ginore attributes: MO MR!477
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=prelu.name)

        assert compare_infer_results_with_caffe(ie_results, prelu.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
