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


def get_tile_params(ie_device=None, precision=None, axis=None, tiles=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param axis: list of set of axis value [(...)]
    :param tiles: list of set of tiles value [(...)]
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
        axis_params = range(1, 4)

    if tiles:
        tiles_params = tiles
    else:
        tiles_params = range(1, 100, 7)

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, axis_params, tiles_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_tile_params)


class TestTile(object):
    @pytest.mark.precommit
    def test_tile_precommit(self, ie_device, precision, axis, tiles):
        self.tile(ie_device, precision, axis, tiles)

    @pytest.mark.nightly
    def test_tile_nightly(self, ie_device, precision, axis, tiles):
        self.tile(ie_device, precision, axis, tiles)

    def tile(self, ie_device, precision, axis, tiles):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        tile = network.add_layer(layer_type='Tile',
                                 inputs=[output],
                                 axis=axis,
                                 tiles=tiles,
                                 get_out_shape_def=caffe_calc_out_shape_tile_layer,
                                 framework_representation_def=tile_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=tile.name)

        assert compare_infer_results_with_caffe(ie_results, tile.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
