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


def get_region_yolo_params(ie_device=None, precision=None, classes=None, coords=None, num=None, axis=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param classes: list of classes values
    :param coords: list of coords values
    :param num: list of num values
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

    if classes:
        classes_type_params = classes
    else:
        classes_type_params = [1, 20]

    if coords:
        coords_params = coords
    else:
        coords_params = [1, 4]

    if num:
        num_params = num
    else:
        num_params = [1, 5]

    if axis:
        axis_params = axis
    else:
        # FIXME: *-9477: [MO Caffe] MO ignores flatten params of RegionYolo layer
        axis_params = [(1, 3)]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, classes_type_params, coords_params, num_params,
                                     axis_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_region_yolo_params)


class TestRegionYolo(object):
    @pytest.mark.skip("Skip in case of memory error")
    @pytest.mark.precommit
    def test_region_yolo_precommit(self, ie_device, precision, classes, coords, num, axis):
        self.region_yolo(ie_device, precision, classes, coords, num, axis)

    @pytest.mark.skip("Skip in case of memory error")
    @pytest.mark.nightly
    def test_region_yolo_nightly(self, ie_device, precision, classes, coords, num, axis):
        self.region_yolo(ie_device, precision, classes, coords, num, axis)

    def region_yolo(self, ie_device, precision, classes, coords, num, axis):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        reg_yolo = network.add_layer(layer_type='RegionYolo',
                                     inputs=[output],
                                     classes=classes,
                                     coords=coords,
                                     num=num,
                                     axis=axis[0],
                                     end_axis=axis[1],
                                     get_out_shape_def=caffe_calc_out_shape_flatten_layer,
                                     framework_representation_def=region_yolo_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=reg_yolo.name)

        assert compare_infer_results_with_caffe(ie_results, reg_yolo.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
