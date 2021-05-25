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


def get_elu_params(ie_device=None, precision=None, alpha=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param alpha: list of set of axis value [(...)]
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if alpha:
        alpha_params = alpha
    else:
        # TODO: add checking default value = 1.0
        alpha_params = range(0, 4)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, alpha_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_elu_params)


class TestELU(object):
    @pytest.mark.precommit
    def test_elu_precommit(self, ie_device, precision, alpha):
        self.elu(ie_device, precision, alpha)

    @pytest.mark.nightly
    def test_elu_nightly(self, ie_device, precision, alpha):
        self.elu(ie_device, precision, alpha)

    def elu(self, ie_device, precision, alpha):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        conv = network.add_layer(layer_type='Convolution',
                                 inputs=[output],
                                 dilations=(1, 1),  # default value
                                 group=1,  # default value
                                 output=10, # random value, used to get negative values in conv output
                                 kernel=(1, 1),
                                 strides=(1, 1),
                                 pads_begin=(0, 0),
                                 pads_end=(0, 0),
                                 get_out_shape_def=caffe_calc_out_shape_conv_layer,
                                 framework_representation_def=conv_to_proto)
        elu = network.add_layer(layer_type='Activation',    # layer_type='Activation' to compare with IE.
                                # In elu_to_proto explicitly used layer_type='ELU' in generation of Caffe model
                                inputs=[conv],
                                alpha=alpha,
                                get_out_shape_def=calc_same_out_shape,
                                type='elu',
                                framework_representation_def=elu_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=elu.name)

        assert compare_infer_results_with_caffe(ie_results, elu.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
