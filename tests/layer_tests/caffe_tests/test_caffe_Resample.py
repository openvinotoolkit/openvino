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


def get_resample_params(ie_device=None, precision=None, resample_type=None, antialias=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param resample_type: list of resample_type values
        only CUBIC, LINEAR and NEAREST interpolation is supported for now
    :param antialias: list of antialias values
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if resample_type:
        resample_type_params = resample_type
    else:
        resample_type_params = ['caffe.ResampleParameter.NEAREST',
                                'caffe.ResampleParameter.LINEAR',
                                'caffe.ResampleParameter.CUBIC']

    if antialias:
        antialias_params = antialias
    else:
        antialias_params = [0, 1]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, resample_type_params, antialias_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_resample_params)


class TestResample(object):
    @pytest.mark.precommit
    def test_resample_precommit(self, ie_device, precision, resample_type, antialias):
        self.resample(ie_device, precision, resample_type, antialias)

    @pytest.mark.nightly
    def test_resample_nightly(self, ie_device, precision, resample_type, antialias):
        self.resample(ie_device, precision, resample_type, antialias)

    def resample(self, ie_device, precision, resample_type, antialias):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        resample = network.add_layer(layer_type='Resample',
                                     inputs=[output],
                                     type=resample_type,
                                     antialias=antialias,
                                     height=224,
                                     width=224,
                                     factor=1.0,  # default value
                                     get_out_shape_def=caffe_calc_out_shape_resample_layer,
                                     framework_representation_def=resample_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=resample.name)

        assert compare_infer_results_with_caffe(ie_results, resample.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
