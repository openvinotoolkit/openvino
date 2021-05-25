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


def get_soft_max_params(ie_device=None, precision=None, axis=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param axis: list of axis values
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
        # TODO: MO support only axis = [1, 2] inference-engine/model-optimizer-tensorflow/blob/master/mo/front/mxnet/extractors/softmax.py
        axis_params = range(-2, 5)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, axis_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_soft_max_params)


class TestSoftMax(object):
    @pytest.mark.skip("Not implemented yet")
    @pytest.mark.precommit
    def test_sigmoid_precommit(self, ie_device, precision, axis):
        self.soft_max_activation(ie_device, precision, axis)

    @pytest.mark.skip("Not implemented yet")
    @pytest.mark.nightly
    def test_soft_max_activation_nightly(self, ie_device, precision, axis):
        self.soft_max_activation(ie_device, precision, axis)

    def soft_max_activation(self, ie_device, precision, axis):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        input_shape = network.get_input_shape()
        # reshape_shape = (1, np.prod(input_shape) // input_shape[1], input_shape[1])
        reshape_shape = (1, np.prod(input_shape))
        reshape = network.add_layer(layer_type='Reshape',
                                    inputs=[inputl],
                                    dim=reshape_shape,
                                    axis=0,  # default value
                                    get_out_shape_def=mxnet_calc_out_shape_reshape_layer,
                                    framework_representation_def=reshape_to_symbol)
        softmax = network.add_layer(layer_type='Softmax',
                                    inputs=[reshape],
                                    axis=axis,
                                    get_out_shape_def=calc_same_out_shape,
                                    framework_representation_def=soft_max_activation_to_symbol)
        network.generate_mxnet_model(mxnet_models_path)
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, skip_layers_types_pattern="Softmax2_flat"), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device, batch=1,
                                 image_path=img_path, out_blob_name=softmax.name + '_flat')

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, softmax.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")

    @pytest.mark.precommit
    def test_soft_max_output_precommit(self, ie_device, precision, axis):
        self.soft_max_output(ie_device, precision, axis)

    @pytest.mark.nightly
    def test_soft_max_output_nightly(self, ie_device, precision, axis):
        self.soft_max_output(ie_device, precision, axis)

    def soft_max_output(self, ie_device, precision, axis):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        flatten = network.add_layer(layer_type='Reshape',
                                    inputs=[inputl],
                                    axis=1,
                                    get_out_shape_def=mxnet_calc_out_shape_flatten_layer,
                                    framework_representation_def=flatten_to_symbol)
        softmax = network.add_layer(layer_type='Softmax',
                                    inputs=[flatten],
                                    # TODO: Check all attributes
                                    # https://mxnet.apache.org/api/python/symbol/symbol.html?highlight=softmaxout#mxnet.symbol.SoftmaxOutput
                                    axis=1,  # default value
                                    get_out_shape_def=calc_same_out_shape,
                                    framework_representation_def=soft_max_output_to_symbol)
        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device, batch=1,
                                 image_path=img_path, out_blob_name=softmax.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, softmax.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
