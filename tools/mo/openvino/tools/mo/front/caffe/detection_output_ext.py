# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.ops.DetectionOutput import DetectionOutput
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class DetectionOutputFrontExtractor(FrontExtractorOp):
    op = 'DetectionOutput'
    enabled = True

    @classmethod
    def extract(cls, node):
        pl = node.pb
        assert pl, 'Protobuf layer can not be empty'

        param = pl.detection_output_param

        # TODO rewrite params as complex structures
        if hasattr(param, 'nms_param'):
            nms_threshold = param.nms_param.nms_threshold
            eta = param.nms_param.eta
            if param.nms_param.top_k == 0:
                top_k = -1
            else:
                top_k = param.nms_param.top_k

        code_type_values = [
            "",
            "caffe.PriorBoxParameter.CORNER",
            "caffe.PriorBoxParameter.CENTER_SIZE",
            "caffe.PriorBoxParameter.CORNER_SIZE"
        ]

        code_type = code_type_values[1]
        if hasattr(param, 'code_type'):
            if param.code_type < 1 or param.code_type > 3:
                log.error("Incorrect value of code_type parameter")
                return
            code_type = code_type_values[param.code_type]

        visualize_threshold = param.visualize_threshold if param.visualize_threshold else 0.6

        resize_mode_values = [
            "",
            "caffe.ResizeParameter.WARP",
            "caffe.ResizeParameter.FIT_SMALL_SIZE",
            "caffe.ResizeParameter.FIT_LARGE_SIZE_AND_PAD"
        ]

        if param.save_output_param.resize_param.resize_mode < 1 or param.save_output_param.resize_param.resize_mode > 3:
            log.error("Incorrect value of resize_mode parameter")
            return
        resize_mode = resize_mode_values[param.save_output_param.resize_param.resize_mode]

        pad_mode_values = [
            "",
            "caffe.ResizeParameter.CONSTANT",
            "caffe.ResizeParameter.MIRRORED",
            "caffe.ResizeParameter.REPEAT_NEAREST"
        ]

        if param.save_output_param.resize_param.pad_mode < 1 or param.save_output_param.resize_param.pad_mode > 3:
            log.error("Incorrect value of pad_mode parameter")
        else:
            pad_mode = pad_mode_values[param.save_output_param.resize_param.pad_mode]

        interp_mode_values = [
            "",
            "caffe.ResizeParameter.LINEAR",
            "caffe.ResizeParameter.AREA",
            "caffe.ResizeParameter.NEAREST",
            "caffe.ResizeParameter.CUBIC",
            "caffe.ResizeParameter.LANCZOS4"
        ]
        interp_mode = ""
        for x in param.save_output_param.resize_param.interp_mode:
            if x < 1 or x > 5:
                log.error("Incorrect value of interp_mode parameter")
                return
            interp_mode += interp_mode_values[x]

        attrs = {
            'share_location': int(param.share_location),
            'background_label_id': param.background_label_id,
            'code_type': code_type,
            'variance_encoded_in_target': int(param.variance_encoded_in_target),
            'keep_top_k': param.keep_top_k,
            'confidence_threshold': param.confidence_threshold,
            'visualize': param.visualize,
            'visualize_threshold': visualize_threshold,
            'save_file': param.save_file,
            # nms_param
            'nms_threshold': nms_threshold,  # pylint: disable=possibly-used-before-assignment
            'top_k': top_k,  # pylint: disable=possibly-used-before-assignment
            'eta': eta,  # pylint: disable=possibly-used-before-assignment
            # save_output_param
            'output_directory': param.save_output_param.output_directory,
            'output_name_prefix': param.save_output_param.output_name_prefix,
            'output_format': param.save_output_param.output_format,
            'label_map_file': param.save_output_param.label_map_file,
            'name_size_file': param.save_output_param.name_size_file,
            'num_test_image': param.save_output_param.num_test_image,
            # save_output_param.resize_param
            'prob': param.save_output_param.resize_param.prob,
            'resize_mode': resize_mode,
            'height': param.save_output_param.resize_param.height,
            'width': param.save_output_param.resize_param.width,
            'height_scale': param.save_output_param.resize_param.height_scale,
            'width_scale': param.save_output_param.resize_param.width_scale,
            'pad_mode': pad_mode,  # pylint: disable=possibly-used-before-assignment
            'pad_value': ','.join(str(x) for x in param.save_output_param.resize_param.pad_value),
            'interp_mode': interp_mode,
        }

        # these params can be omitted in caffe.proto and in param as consequence,
        # so check if it is set or set to default
        fields = [field[0].name for field in param.ListFields()]
        if 'input_width' in fields:
            attrs['input_width'] = param.input_width
        if 'input_height' in fields:
            attrs['input_height'] = param.input_height
        if 'normalized' in fields:
            attrs['normalized'] = int(param.normalized)
        if 'objectness_score' in fields:
            attrs['objectness_score'] = param.objectness_score

        mapping_rule = merge_attrs(param, attrs)

        # update the attributes of the node
        DetectionOutput.update_node_stat(node, mapping_rule)
        return cls.enabled
