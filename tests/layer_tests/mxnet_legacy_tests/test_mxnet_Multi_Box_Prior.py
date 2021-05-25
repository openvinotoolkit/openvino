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


def get_multi_box_prior_params(ie_device=None, precision=None, step=None, min_size=None, max_size=None, flip=None,
                               clip=None):
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
        step_params = list(np.arange(0.03, 0.06, 0.01))

    if min_size:
        min_params = min_size
    else:
        min_params = list(np.arange(0.1, 0.15, 0.015))

    if max_size:
        max_params = max_size
    else:
        max_params = list(np.arange(0.15, 0.2, 0.015))

    if flip:
        flip_params = flip
    else:
        flip_params = [0]

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


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_multi_box_prior_params)


class TestMultiBoxPrior(object):
    @pytest.mark.precommit
    def test_multi_box_prior_precommit(self, ie_device, precision, step, min_size, max_size, flip, clip):
        self.multi_box_prior(ie_device, precision, step, min_size, max_size, flip, clip)

    @pytest.mark.nightly
    def test_multi_box_prior_nightly(self, ie_device, precision, step, min_size, max_size, flip, clip):
        self.multi_box_prior(ie_device, precision, step, min_size, max_size, flip, clip)

    def multi_box_prior(self, ie_device, precision, step, min_size, max_size, flip, clip):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[inputl],
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_symbol)
        priorbox = network.add_layer(layer_type='PriorBox',
                                     inputs=[relu, inputl],
                                     step=step,
                                     min_size=[min_size, max_size],
                                     max_size=None,
                                     scale_all_sizes=0,
                                     offset=0.5,
                                     aspect_ratio=(1, 2, 0.5),
                                     variance="0.100000,0.100000,0.200000,0.200000",
                                     flip=flip,
                                     clip=clip,
                                     img_size=0,  # default value
                                     img_h=0,  # default value
                                     img_w=0,  # default value
                                     step_h=0,  # default value
                                     step_w=0,  # default value
                                     get_out_shape_def=mxnet_calc_out_shape_multi_box_prior_layer,
                                     framework_representation_def=multi_box_prior_to_symbol)

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
                                 image_path=img_path, out_blob_name=priorbox.name)
        ie_results = np.reshape(ie_results, np.prod(ie_results.shape))
        ie_results = ie_results[: ie_results.size // 2]
        ie_results = np.reshape(ie_results, (np.prod(ie_results.shape) // 4, 4))

        assert compare_infer_results_with_mxnet(
            ie_results, network.name, priorbox.name, input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
