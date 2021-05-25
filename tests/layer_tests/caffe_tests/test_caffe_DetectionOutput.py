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


def get_detection_output_params(ie_device=None, precision=None, num_classes=None, background_label_id=None, top_k=None,
                                variance_encoded_in_target=None, keep_top_k=None, num_orient_classes=None,
                                code_type=None, share_location=None, interpolate_orientation=None,
                                nms_threshold=None, confidence_threshold=None):
    """
    Name: DetectionOutput
    Short description: DetectionOutput layer performs non-maximum suppression to generate the detection output using
    information on location and confidence predictions.

    Parameters
    DetectionOutput layer parameters should be specified as the data node, which is placed as a child of the layer node.

    :param num_classes

          Description: number of classes to be predicted
          Range of values: positive integer values

    :param background_label_id

          Description: background label id. If there is no background class, set it to -1.
          Range of values: integer values

    :param top_k

          Description: maximum number of results to be kept on NMS stage
          Range of values: integer values

    :param variance_encoded_in_target

          Description: if true, variance is encoded in target; otherwise we need to adjust the predicted offset accordingly.
          Range of values: logical values

    :param keep_top_k

          Description: number of total bboxes to be kept per image after NMS step.-1 means keeping all bboxes after NMS step
          Range of values: integer values

    :param num_orient_classes

          Range of values: integer values

    :param code_type

          Description: type of coding method for bounding boxes
          Range of values: caffe.PriorBoxParameter.CENTER_SIZE and others

    :param share_location

          Description: bounding boxes are shared among different classes
          Range of values: logical values

    :param interpolate_orientation

          Range of values: integer values

    :param nms_threshold

          Description: threshold to be used in NMS stage.
          Range of values: floating point values

    :param confidence_threshold

          Description: only consider detections whose confidences are larger than a threshold.
                       If not provided, consider all boxes.
          Range of values: floating point values

    :param ie_device: list of devices to be scored with

    :param precision: list of precisions
    """
    if ie_device:
        ie_device_params = ie_device
    else:
        ie_device_params = test_device

    if precision:
        precision_params = precision
    else:
        precision_params = test_precision

    if num_classes:
        num_classes_params = num_classes
    else:
        num_classes_params = [4]

    if background_label_id:
        background_label_id_params = background_label_id
    else:
        background_label_id_params = [0]

    if top_k:
        top_k_params = top_k
    else:
        top_k_params = [400]

    if variance_encoded_in_target:
        variance_encoded_in_target_params = variance_encoded_in_target
    else:
        variance_encoded_in_target_params = [1]

    if keep_top_k:
        keep_top_k_params = keep_top_k
    else:
        keep_top_k_params = [200]

    if num_orient_classes:
        num_orient_classes_params = num_orient_classes
    else:
        num_orient_classes_params = [5]

    if code_type:
        code_type_params = code_type
    else:
        code_type_params = ['caffe.PriorBoxParameter.CORNER']

    if share_location:
        share_location_params = share_location
    else:
        share_location_params = [1]

    if interpolate_orientation:
        interpolate_orientation_params = interpolate_orientation
    else:
        interpolate_orientation_params = [1]

    if nms_threshold:
        nms_threshold_params = nms_threshold
    else:
        nms_threshold_params = [0.45]

    if confidence_threshold:
        confidence_threshold_params = confidence_threshold
    else:
        confidence_threshold_params = [0.01]

    test_args = []
    for element in itertools.product(ie_device_params, precision_params, num_classes_params, background_label_id_params,
                                     top_k_params,
                                     variance_encoded_in_target_params, keep_top_k_params, num_orient_classes_params,
                                     code_type_params, share_location_params, interpolate_orientation_params,
                                     nms_threshold_params, confidence_threshold_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)

    return test_args


def pytest_generate_tests(metafunc):
    generate_tests(metafunc, get_detection_output_params)


class TestDetectionOutput(object):
    @pytest.mark.precommit
    def test_detection_output_precommit(self, ie_device, precision, num_classes, background_label_id, top_k,
                                        variance_encoded_in_target,
                                        keep_top_k, num_orient_classes, code_type, share_location,
                                        interpolate_orientation,
                                        nms_threshold, confidence_threshold):
        self.detection_output(ie_device, precision, num_classes, background_label_id, top_k,
                              variance_encoded_in_target,
                              keep_top_k, num_orient_classes, code_type, share_location, interpolate_orientation,
                              nms_threshold, confidence_threshold)

    @pytest.mark.nightly
    def test_detection_output_nightly(self, ie_device, precision, num_classes, background_label_id, top_k,
                                      variance_encoded_in_target,
                                      keep_top_k, num_orient_classes, code_type, share_location,
                                      interpolate_orientation,
                                      nms_threshold, confidence_threshold):
        self.detection_output(ie_device, precision, num_classes, background_label_id, top_k,
                              variance_encoded_in_target,
                              keep_top_k, num_orient_classes, code_type, share_location, interpolate_orientation,
                              nms_threshold, confidence_threshold)

    def detection_output(self, ie_device, precision, num_classes, background_label_id, top_k,
                         variance_encoded_in_target,
                         keep_top_k, num_orient_classes, code_type, share_location, interpolate_orientation,
                         nms_threshold, confidence_threshold):
        network = Net(precision=precision)
        output = network.add_layer(layer_type='Input',
                                   get_out_shape_def=calc_out_shape_input_layer,
                                   framework_representation_def=input_to_proto)
        flatten1 = network.add_layer(layer_type='Flatten',
                                     inputs=[output],
                                     axis=1,
                                     end_axis=3,
                                     get_out_shape_def=caffe_calc_out_shape_flatten_layer,
                                     framework_representation_def=flatten_to_proto)
        relu = network.add_layer(layer_type='ReLU',
                                 inputs=[output],
                                 negative_slope=0,
                                 engine='caffe.ReLUParameter.DEFAULT',
                                 get_out_shape_def=calc_same_out_shape,
                                 framework_representation_def=relu_to_proto)
        flatten2 = network.add_layer(layer_type='Flatten',
                                     inputs=[relu],
                                     axis=1,
                                     end_axis=3,
                                     get_out_shape_def=caffe_calc_out_shape_flatten_layer,
                                     framework_representation_def=flatten_to_proto)
        flatten3 = network.add_layer(layer_type='Flatten',
                                     inputs=[output],
                                     axis=2,
                                     end_axis=3,
                                     get_out_shape_def=caffe_calc_out_shape_flatten_layer,
                                     framework_representation_def=flatten_to_proto)
        tile = network.add_layer(layer_type='Tile',
                                 inputs=[flatten3],
                                 axis=2,
                                 tiles=3,
                                 get_out_shape_def=caffe_calc_out_shape_tile_layer,
                                 framework_representation_def=tile_to_proto)
        do = network.add_layer(layer_type='DetectionOutput',
                               inputs=[flatten1, flatten2, tile],
                               num_classes=num_classes,
                               background_label_id=background_label_id,
                               top_k=top_k,
                               eta=0.5,
                               input_height=3,
                               input_width=4,
                               visualize="False",  # default
                               variance_encoded_in_target=variance_encoded_in_target,
                               normalized=1,
                               keep_top_k=keep_top_k,
                               code_type=code_type,
                               share_location=share_location,
                               nms_threshold=nms_threshold,
                               confidence_threshold=confidence_threshold,
                               get_out_shape_def=caffe_calc_out_shape_detection_output_layer,
                               framework_representation_def=detection_output_to_proto)
        network.save_caffemodel(caffe_models_path)
        generate_ir_from_caffe(name=network.name, precision=precision)
        ir_model = IR(os.path.join(ir_path, network.name + '.xml'))
        network_cur = Net(precision=precision)
        network_cur.load_xml(ir_model.xml)
        network_cur.load_bin(os.path.join(ir_path, network.name + '.bin'))

        assert network.compare(network_cur, ignore_attributes={'ReLU': ['engine', 'negative_slope']}), \
            "Comparing of networks failed."
        lg.info("[ TEST ] Comparing of networks passed")

        ie_results = score_model(model_path=os.path.join(ir_path, network.name + '.xml'), device=ie_device,
                                 image_path=img_path, out_blob_name=do.name)

        assert compare_infer_results_with_caffe(ie_results, do.name), "Comparing with Caffe failed."
        lg.info("[ TEST ] Comparing of scoring results passed")
