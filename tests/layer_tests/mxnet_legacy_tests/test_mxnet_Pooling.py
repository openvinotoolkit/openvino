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


def get_pool_params(ie_device=None, precision=None, kernel=None, pad=None, stride=None, method=None,
                    convention=None, global_pool=None):
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if kernel:
        kernel_params = kernel
    else:
        kernel_params = [tuple(np.multiply((1, 1), u)) for u in range(3, 10)]

    if pad:
        padding_params = pad
    else:
        padding_params = [tuple(np.multiply(np.ones(2, int), u)) for u in range(1, 4)]

    if stride:
        stride_params = stride
    else:
        stride_params = [tuple(np.multiply((1, 1), u)) for u in range(1, 4)]

    if method:
        method_params = method
    else:
        # FIXME: IE doesn't support "sum" operation
        method_params = ['max', 'avg']
        # method_params = ['max', 'avg', 'sum']

    if convention:
        convention_params = convention
    else:
        convention_params = ['full', 'valid']

    if global_pool:
        global_pool_params = global_pool
    else:
        # FIXME: *-9854 MO doesn't support "int" values
        # global_pool_params = [True, False, 1, 0]
        # global_pool_params = [True, False]
        # FIXME: global_pool=True not supported now. Need to reshape output shape
        global_pool_params = [False, 0]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params,
                                     kernel_params, padding_params, stride_params, method_params,
                                     convention_params, global_pool_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    scope_for_marker = {
        "precommit": dict(
            kernel=[(1, 1), (1, 3)],
            pad=[(1, 1), (1, 3)],
            stride=[(1, 1), (1, 3)],
            global_pool=[0]
        )}
    generate_tests(metafunc, get_pool_params, **scope_for_marker)


class TestPooling(object):
    @pytest.mark.precommit
    def test_pool_precommit(self, ie_device, precision, kernel, pad, stride, method, convention, global_pool):
        self.pool(ie_device, precision, kernel, pad, stride, method, convention, global_pool)

    @pytest.mark.nightly
    def test_pool_nightly(self, ie_device, precision, kernel, pad, stride, method, convention, global_pool):
        self.pool(ie_device, precision, kernel, pad, stride, method, convention, global_pool)

    def pool(self, ie_device, precision, kernel, pad, stride, method, convention, global_pool):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        pool = network.add_layer(layer_type='Pooling',
                                 inputs=[inputl],
                                 pool_method=method,
                                 kernel=kernel,
                                 strides=stride,
                                 pads_begin=pad,
                                 pads_end=pad,
                                 convention=convention,
                                 global_pool=global_pool,  # ignored parameter
                                 exclude_pad="false",  # ignored parameter
                                 get_out_shape_def=mxnet_calc_out_shape_pooling_layer,
                                 framework_representation_def=pool_to_symbol)
        network.generate_mxnet_model(mxnet_models_path)
        input_shape = network.get_input_shape()
        generate_ir_from_mxnet(name=network.name, input_shape=input_shape,
                               input_names=[inputl.name], precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=pool.name)

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, pool.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
