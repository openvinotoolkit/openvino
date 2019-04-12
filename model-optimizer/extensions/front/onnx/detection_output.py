"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.op import Op
from mo.utils.error import Error


class DetectionOutputFrontExtractor(FrontExtractorOp):
    op = 'DetectionOutput'
    enabled = True

    @staticmethod
    def extract(node):
        nms_threshold = onnx_attr(node, 'nms_threshold', 'f', default=0.0)
        eta = onnx_attr(node, 'eta', 'f', default=0.0)
        top_k = onnx_attr(node, 'top_k', 'i', default=-1)

        code_type_values = {
            b"CORNER": "caffe.PriorBoxParameter.CORNER",
            b"CENTER_SIZE": "caffe.PriorBoxParameter.CENTER_SIZE",
        }

        code_type = onnx_attr(node, 'code_type', 's', default=code_type_values[b"CORNER"])
        try:
            code_type = code_type_values[code_type]
        except KeyError:
            raise Error("Incorrect value of code_type parameter {}".format(code_type))

        resize_mode_values = {
            b"": "",
            b"WARP": "caffe.ResizeParameter.WARP",
            b"FIT_SMALL_SIZE": "caffe.ResizeParameter.FIT_SMALL_SIZE",
            b"FIT_LARGE_SIZE_AND_PAD": "caffe.ResizeParameter.FIT_LARGE_SIZE_AND_PAD",
        }
        resize_mode = onnx_attr(node, 'resize_mode', 's', default=b"")
        try:
            resize_mode = resize_mode_values[resize_mode]
        except KeyError:
            raise Error("Incorrect value of resize_mode parameter {}".format(resize_mode))

        pad_mode_values = {
            b"": "",
            b"CONSTANT": "caffe.ResizeParameter.CONSTANT",
            b"MIRRORED": "caffe.ResizeParameter.MIRRORED",
            b"REPEAT_NEAREST": "caffe.ResizeParameter.REPEAT_NEAREST"
        }
        pad_mode = onnx_attr(node, 'pad_mode', 's', default=b"")
        try:
            pad_mode = pad_mode_values[pad_mode]
        except KeyError:
            raise Error("Incorrect value of pad_mode parameter {}".format(pad_mode))

        interp_mode_values = {
            b"": "",
            b"LINEAR": "caffe.ResizeParameter.LINEAR",
            b"AREA": "caffe.ResizeParameter.AREA",
            b"NEAREST": "caffe.ResizeParameter.NEAREST",
            b"CUBIC": "caffe.ResizeParameter.CUBIC",
            b"LANCZOS4": "caffe.ResizeParameter.LANCZOS4"
        }
        interp_mode = onnx_attr(node, 'interp_mode', 's', default=b"")
        try:
            interp_mode = interp_mode_values[interp_mode]
        except KeyError:
            raise Error("Incorrect value of interp_mode parameter {}".format(interp_mode))

        attrs = {
            'num_classes': onnx_attr(node, 'num_classes', 'i', default=0),
            'share_location': onnx_attr(node, 'share_location', 'i', default=0),
            'background_label_id': onnx_attr(node, 'background_label_id', 'i', default=0),
            'code_type': code_type,
            'variance_encoded_in_target': onnx_attr(node, 'variance_encoded_in_target', 'i', default=0),
            'keep_top_k': onnx_attr(node, 'keep_top_k', 'i', default=0),
            'confidence_threshold':  onnx_attr(node, 'confidence_threshold', 'f', default=0),
            'visualize_threshold': onnx_attr(node, 'visualize_threshold', 'f', default=0.6),
            # nms_param
            'nms_threshold': nms_threshold,
            'top_k': top_k,
            'eta': eta,
            # save_output_param.resize_param
            'prob': onnx_attr(node, 'prob', 'f', default=0),
            'resize_mode': resize_mode,
            'height': onnx_attr(node, 'height', 'i', default=0),
            'width': onnx_attr(node, 'width', 'i', default=0),
            'height_scale': onnx_attr(node, 'height_scale', 'i', default=0),
            'width_scale': onnx_attr(node, 'width_scale', 'i', default=0),
            'pad_mode': pad_mode,
            'pad_value': onnx_attr(node, 'pad_value', 's', default=""),
            'interp_mode': interp_mode,
            'input_width': onnx_attr(node, 'input_width', 'i', default=1),
            'input_height': onnx_attr(node, 'input_height', 'i', default=1),
            'normalized': onnx_attr(node, 'normalized', 'i', default=1),
        }

        # update the attributes of the node
        Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)
        return __class__.enabled
