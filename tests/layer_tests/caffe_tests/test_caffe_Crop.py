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


def get_crop_params(ie_device=None, precision=None, axis=None, offset=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param axis: list of set of axis value [(...)]
    :param offset: list of set of offset value [(...)]
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
        axis_params = range(-1, 4)

    if offset:
        offset_params = offset
    else:
        offset_params = [0]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, axis_params, offset_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        if element[0] == 'MYRIAD':
            if element[1] == 'FP32':
                continue
            if (element[2], element[3]) in [(-1, 0), (0, 0)]:
                continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_crop_params)


def get_network(precision, axis, off):
    network = Net(precision=precision)
    ax = ['0', '1', '2', '3']
    offset = [str(off)] * 4
    output1 = network.add_layer(layer_type='Input',
                                get_out_shape_def=calc_out_shape_input_layer,
                                framework_representation_def=input_to_proto)
    output2 = network.add_layer(layer_type='Input',
                                get_out_shape_def=calc_out_shape_input_layer,
                                framework_representation_def=input_to_proto)
    network.add_layer(layer_type='Crop',
                      inputs=[output1, output2],
                      offset=','.join(offset[axis:]),
                      axis=','.join(ax[axis:]),
                      get_out_shape_def=caffe_calc_out_shape_crop_layer,
                      framework_representation_def=crop_to_proto)
    return network


class TestCrop(object):
    @pytest.mark.precommit
    def test_crop_precommit(self, ie_device, precision, axis, offset):
        self.crop(ie_device, precision, axis, offset)

    @pytest.mark.nightly
    def test_crop_nightly(self, ie_device, precision, axis, offset):
        self.crop(ie_device, precision, axis, offset)

    def crop(self, ie_device, precision, axis, offset):
        network = get_network(precision, axis, offset)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision, disable_fusing=True)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='Crop2')

        assert compare_infer_results_with_caffe(ie_results, 'Crop2'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
