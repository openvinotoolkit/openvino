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


def get_flatten_params(ie_device=None, precision=None, axis=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param axis:  list of pair: axis and end_axis values
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
        axis_params = [1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, axis_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    # TODO: Copied from caffe flatten. Check it
    """
    for device in ie_device_params:
        if device in ['CPU', 'GPU', 'MYRIAD']:
            # CPU, GPU and MYRIAD don't support 'C' layout
            test_args.remove((device, (0, 3)))
        if device == 'GPU':
            # GPU doesn't support 'CHW' layout
            test_args.remove((device, (0, 1)))
            test_args.remove((device, (1, 2)))
            test_args.remove((device, (2, 3)))
    """
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_flatten_params)


class TestFlatten(object):
    @pytest.mark.precommit
    def test_flatten_precommit(self, ie_device, precision, axis):
        self.flatten(ie_device, precision, axis)

    @pytest.mark.nightly
    def test_flatten_nightly(self, ie_device, precision, axis):
        self.flatten(ie_device, precision, axis)

    def flatten(self, ie_device, precision, axis):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        flatten = network.add_layer(layer_type='Reshape',
                                    inputs=[inputl],
                                    axis=axis,
                                    get_out_shape_def=mxnet_calc_out_shape_flatten_layer,
                                    framework_representation_def=flatten_to_symbol)

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

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=flatten.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, flatten.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
