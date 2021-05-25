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


def get_st_params(ie_device=None, precision=None, transform_type=None, sampler_type=None, to_compute_du=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param transform_type: list if transform_type values
        Transformation type only supports affine now!
    :param sampler_type: list if sampler_type values
        Sampler type only supports bilinear now!
    :param to_compute_du: list if to_compile_du values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if transform_type:
        transform_params = transform_type
    else:
        transform_params = ['affine']

    if sampler_type:
        sampler_params = sampler_type
    else:
        sampler_params = ['bilinear']

    if to_compute_du:
        to_compute_du_params = to_compute_du
    else:
        to_compute_du_params = [0, 1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, transform_params, sampler_params,
                                     to_compute_du_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def get_st_with_output_params(ie_device=None, precision=None, transform_type=None, sampler_type=None,
                              to_compute_du=None,
                              output_size=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param transform_type: list if transform_type values
        Transformation type only supports affine now!
    :param sampler_type: list if sampler_type values
        Sampler type only supports bilinear now!
    :param to_compute_du: list if to_compile_du values
    :param output_size: list if output_size values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if transform_type:
        transform_params = transform_type
    else:
        transform_params = ['affine']

    if sampler_type:
        sampler_params = sampler_type
    else:
        sampler_params = ['bilinear']

    if to_compute_du:
        to_compute_du_params = to_compute_du
    else:
        to_compute_du_params = [0, 1]

    if output_size:
        output_size_params = output_size
    else:
        output_size_params = [tuple(np.multiply(np.ones(2, int), u)) for u in [200, 222, 224, 226, 228]]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, transform_params, sampler_params,
                                     to_compute_du_params,
                                     output_size_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    #   Generate specifiс test suite for every test
    test_func = metafunc.function.__name__
    if "st" in test_func and "output" not in test_func:
        generate_tests(metafunc, get_st_params)
    elif "st_with_output" in test_func:
        generate_tests(metafunc, get_st_with_output_params)


def get_network(precision, transform, sampler, to_compile_du, output_):
    network = Net(precision=precision)
    output = network.add_layer(layer_type='Input',
                               get_out_shape_def=calc_out_shape_input_layer,
                               framework_representation_def=input_to_proto)
    theta = network.add_layer(layer_type='Input',
                              layer_name='Theta',
                              get_out_shape_def=calc_out_shape_theta_layer,
                              framework_representation_def=input_to_proto)
    if not output_:
        output_ = output.outputs['Input0'][0][1][2], output.outputs['Input0'][0][1][3]
    network.add_layer(layer_type='SpatialTransformer',
                      inputs=[output, theta],
                      transform_type=transform,
                      sampler_type=sampler,
                      to_compute_dU=to_compile_du,
                      output_W=output_[0],
                      output_H=output_[1],
                      get_out_shape_def=caffe_calc_out_shape_st_layer,
                      framework_representation_def=st_to_proto)
    return network


class TestST(object):
    @pytest.mark.precommit
    def test_st_precommit(self, ie_device, precision, transform_type, sampler_type, to_compute_du):
        self.spatial_transformer(ie_device, precision, transform_type, sampler_type, to_compute_du)

    @pytest.mark.nightly
    def test_st_nightly(self, ie_device, precision, transform_type, sampler_type, to_compute_du):
        self.spatial_transformer(ie_device, precision, transform_type, sampler_type, to_compute_du)

    @pytest.mark.precommit
    def test_st_with_output_precommit(self, ie_device, precision, transform_type, sampler_type, to_compute_du,
                                      output_size):
        self.spatial_transformer(ie_device, precision, transform_type, sampler_type, to_compute_du, output_size)

    @pytest.mark.nightly
    def test_st_with_output_nightly(self, ie_device, precision, transform_type, sampler_type, to_compute_du,
                                    output_size):
        self.spatial_transformer(ie_device, precision, transform_type, sampler_type, to_compute_du, output_size)

    def spatial_transformer(self, ie_device, precision, transform_type, sampler_type, to_compute_du, output_size=None):
        network = get_network(precision, transform_type, sampler_type, to_compute_du, output_size)
        network.save_caffemodel(caffe_models_path, 'Input0')
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='SpatialTransformer2')

        assert compare_infer_results_with_caffe(ie_results, 'SpatialTransformer2'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
