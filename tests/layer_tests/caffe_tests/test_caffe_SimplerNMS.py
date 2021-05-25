import itertools
import logging as lg

import pytest
from caffe_tests.conftest import generate_tests
from common.caffe_layers_representation import *
from common.call_ModelOptimizer import generate_ir_from_caffe
from common.constants import *
from common.infer_shapes import *
from common.legacy.generic_ir_comparator import *


def get_simpler_nms_params(ie_device=None, precision=None, nms_topn=None, bbox_size=None, scale=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    :param nms_topn: list of set of nms_topn value [(...)]
    :param scale: list of set of scale value [(...)]
    :param bbox_size: list of set of bbox_size value [(...)]
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if nms_topn:
        nms_topn_params = nms_topn
    else:
        nms_topn_params = range(1, 25, 5)

    if bbox_size:
        bbox_size_params = bbox_size
    else:
        bbox_size_params = range(1, 50, 7)

    if scale:
        scale_params = scale
    else:
        scale_params = [(8.0, 16.0, 32.0)]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, nms_topn_params, bbox_size_params,
                                     scale_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_simpler_nms_params)


class TestSimplerNMS(object):
    @pytest.mark.skip("*-9398: [MO Caffe] Cannot convert net with SimplerNMS layer")
    @pytest.mark.precommit
    def test_simpler_nms_precommit(self, ie_device, precision, nms_topn, bbox_size, scale):
        self.simpler_nms(ie_device, precision, nms_topn, bbox_size, scale)

    @pytest.mark.skip("*-9398: [MO Caffe] Cannot convert net with SimplerNMS layer")
    @pytest.mark.nightly
    def test_simpler_nms_nightly(self, ie_device, precision, nms_topn, bbox_size, scale):
        self.simpler_nms(ie_device, precision, nms_topn, bbox_size, scale)

    def simpler_nms(self, ie_device, precision, nms_topn, bbox_size, scale):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[output],
                                 negative_slope=1,
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_proto)
        fc = network.add_layer(layer_type='FullyConnected',
                               inputs=[output],
                               out_size=3,
                               get_out_shape_def=caffe_calc_out_shape_fullyconnected_layer,
                               framework_representation_def=fullyconnected_to_proto)
        network.add_layer(layer_type='SimplerNMS',
                          inputs=[output, relu, fc],
                          pre_nms_topn=nms_topn,
                          post_nms_topn=nms_topn,
                          cls_threshold=0.5,
                          iou_threshold=0.7,
                          # SimplerNMS layer currently doesn't support other feat_stride value than 16.
                          feat_stride=16,
                          min_bbox_size=bbox_size,
                          scale=scale,
                          get_out_shape_def=caffe_calc_out_shape_simpler_nms_layer,
                          framework_representation_def=simpler_nms_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes=ignore_attributes), "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        # TODO: Resolve problem with scoring
        """
        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name='SimplerNMS3')

        assert compare_infer_results_with_caffe(ie_results, 'SimplerNMS3'), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
        """
