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


def get_up_sampling_params(ie_device=None, precision=None, scale=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param scale: range of scale values
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
        scale_params = range(1, 4)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, scale_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_up_sampling_params)


class TestUpSampling(object):
    @pytest.mark.precommit
    def test_up_sampling_precommit(self, ie_device, precision, scale):
        self.up_sampling(ie_device, precision, scale)

    @pytest.mark.nightly
    def test_up_sampling_nightly(self, ie_device, precision, scale):
        self.up_sampling(ie_device, precision, scale)

    def up_sampling(self, ie_device, precision, scale):
        network = Net(precision=precision)
        inputl = network.add_layer(layer_type='Input',
                                   layer_name="data",
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_symbol)
        upsampling = network.add_layer(layer_type='Resample',
                                       layer_name='upsampling0',
                                       inputs=[inputl],
                                       factor=scale,
                                       antialias=0,  # default value
                                       type='caffe.ResampleParameter.NEAREST',  # default value
                                       get_out_shape_def=mxnet_calc_out_shape_up_sampling_layer,
                                       framework_representation_def=up_sampling_to_symbol)

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
                                 image_path=img_path, out_blob_name=upsampling.name)

        assert compare_infer_results_with_mxnet(ie_results, network.name, upsampling.name,
                                                input_shape), "Comparing with MxNet failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
