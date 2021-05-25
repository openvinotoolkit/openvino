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


def get_roi_pool_params(ie_device=None, precision=None, pool=None, scale=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param pool: list of pool values
    :param scale: list of scale values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if pool:
        pool_params = pool
    else:
        pool_params = [tuple(np.multiply((1, 1), u)) for u in range(3, 10)]

    if scale:
        scale_params = scale
    else:
        scale_params = [0.062500, 0.125]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, pool_params, scale_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def get_ps_roi_pool_params(ie_device=None, precision=None, scale=None, output_dim=None, group_size=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param scale: list of scale values
    :param output_dim: list of output_dim values. Must be more that 0
    :param group_size: list of group_size values. Must be more that 0
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if scale:
        scale_params = scale
    else:
        scale_params = [0.062500, 0.125]

    if output_dim:
        output_dim_params = output_dim
    else:
        output_dim_params = [3]

    if group_size:
        group_size_params = group_size
    else:
        group_size_params = [1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, scale_params, output_dim_params,
                                     group_size_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "roi_pooling" in test_func and "ps" not in test_func:
        generate_tests(metafunc, get_roi_pool_params)
    elif "ps_roi_pooling" in test_func:
        generate_tests(metafunc, get_ps_roi_pool_params)


class TestROIPooling(object):
    @pytest.mark.precommit
    def test_roi_pooling_precommit(self, ie_device, precision, pool, scale):
        self.roi_pooling(ie_device, precision, pool, scale)

    @pytest.mark.nightly
    def test_roi_pooling_nightly(self, ie_device, precision, pool, scale):
        self.roi_pooling(ie_device, precision, pool, scale)

    def roi_pooling(self, ie_device, precision, pool, scale):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        fc = network.add_layer(layer_type='FullyConnected',
                               inputs=[output],
                               out_size=3,
                               get_out_shape_def=caffe_calc_out_shape_fullyconnected_layer,
                               framework_representation_def=fullyconnected_to_proto)
        roipool = network.add_layer(layer_type='ROIPooling',
                                    inputs=[output, fc],
                                    pooled_w=pool[0],
                                    pooled_h=pool[1],
                                    spatial_scale=scale,
                                    get_out_shape_def=caffe_calc_out_shape_roi_pool_layer,
                                    framework_representation_def=roi_pool_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=roipool.name)

        assert compare_infer_results_with_caffe(ie_results, roipool.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_ps_roi_pooling_precommit(self, ie_device, precision, scale, output_dim, group_size):
        self.ps_roi_pooling(ie_device, precision, scale, output_dim, group_size)

    @pytest.mark.nightly
    def test_ps_roi_pooling_nightly(self, ie_device, precision, scale, output_dim, group_size):
        self.ps_roi_pooling(ie_device, precision, scale, output_dim, group_size)

    def ps_roi_pooling(self, ie_device, precision, scale, output_dim, group_size):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        fc = network.add_layer(layer_type='FullyConnected',
                               inputs=[output],
                               out_size=3,
                               get_out_shape_def=caffe_calc_out_shape_fullyconnected_layer,
                               framework_representation_def=fullyconnected_to_proto)
        psroipool = network.add_layer(layer_type='PSROIPooling',
                                      inputs=[output, fc],
                                      spatial_scale=scale,
                                      output_dim=output_dim,
                                      group_size=group_size,
                                      get_out_shape_def=caffe_calc_out_shape_ps_roi_pool_layer,
                                      framework_representation_def=ps_roi_pool_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=psroipool.name)

        assert compare_infer_results_with_caffe(ie_results, psroipool.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
