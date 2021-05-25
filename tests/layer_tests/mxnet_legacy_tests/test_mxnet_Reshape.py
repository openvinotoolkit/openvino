import itertools
import logging as lg

import pytest
from common.call_InferenceEngine import score_model, compare_infer_results_with_mxnet
from common.call_ModelOptimizer import generate_ir_from_mxnet
from common.constants import *
from common.infer_shapes import *
from common.mxnet_layers_representation import *
from common.legacy.generic_ir_comparator import *
from mxnet_legacy_tests.conftest import generate_tests


def get_reshape_params(ie_device=None, precision=None, shape=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param shape: list of sizes
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if shape:
        shape_params = shape
    else:
        shape_params = []
        for s in [(1, 1, 224, 224),
                  (2, 2, 112, 112),
                  (4, 4, 56, 56),
                  (7, 7, 32, 32),
                  (14, 14, 16, 16),
                  (28, 28, 8, 8),
                  (56, 56, 4, 4),
                  (112, 112, 2, 2),
                  (224, 224, 1, 1)]:
            shape_params.append(tuple([1, 3, s[0] * s[1], s[2] * s[3]]))
        # add Flattening
        shape_params.extend([tuple([1, 150528]), tuple([1, 3, 50176])])

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, shape_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_reshape_params)


class TestReshape(object):
    @pytest.mark.precommit
    def test_reshape_precommit(self, ie_device, precision, shape):
        self.reshape(ie_device, precision, shape)

    @pytest.mark.nightly
    def test_reshape_nightly(self, ie_device, precision, shape):
        self.reshape(ie_device, precision, shape)

    def reshape(self, ie_device, precision, shape):
        # Create network for generation of MXNet network
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        reshape = network.add_layer(layer_type='Reshape',
                                    inputs=[inputl],
                                    dim=shape,
                                    get_out_shape_def=mxnet_calc_out_shape_reshape_layer,
                                    framework_representation_def=reshape_to_symbol)
        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        # Create correct network for comparison with IR
        network_to_compare = Net(precision=precision)
        inputl_to_compare = network_to_compare.add_layer(layer_type='Input',
                                                         layer_name="data",
                                                         get_out_shape_def=calc_out_shape_input_layer,
                                                         framework_representation_def=input_to_symbol)
        const = network_to_compare.add_layer(layer_type='Const',
                                             get_out_shape_def=lambda x: len(shape))
        network_to_compare.add_layer(layer_type='Reshape',
                                     inputs=[inputl_to_compare, const],
                                     dim=shape,
                                     get_out_shape_def=mxnet_calc_out_shape_reshape_layer,
                                     framework_representation_def=reshape_to_symbol)

        assert network_to_compare.compare(network_cur, ignore_attributes={"Reshape": ["dim"]}), \
            "Comparing of networks failed."
        del network_to_compare
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=reshape.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, reshape.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
