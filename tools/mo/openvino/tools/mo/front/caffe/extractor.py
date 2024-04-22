# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.caffe.extractors.native_caffe import native_caffe_node_extractor
from openvino.tools.mo.front.common.register_custom_ops import extension_op_extractor
from openvino.tools.mo.front.extractor import CaffePythonFrontExtractorOp
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def node_pb_arg(pb_extractor):
    return lambda node: pb_extractor(node.pb, node.model_pb)


"""
Keys are names that appear as layer names in .prototxt.
Full list is available here: http://caffe.berkeleyvision.org/tutorial/layers.html
"""

caffe_type_extractors = {}


def common_caffe_fields(node: Node) -> dict:
    if node.has_valid('op') and node.op == 'Identity':
        return {}
    pb = node.pb if node.pb else node
    layer_type = pb.type
    if isinstance(layer_type, int):
        layer_type = pb.LayerType.DESCRIPTOR.values_by_number[layer_type].name
    layer_type = str(layer_type)

    return {
        'kind': 'op',
        'name': pb.name,
        'type': layer_type,
        'op': layer_type,
        # generic code relies on op; it should be overridden by specific op extractor
        'infer': None,
    }


def caffe_extractor(node: Node, lowered_keys_map: dict) -> (bool, dict):
    if node.has_valid('op') and node.op == 'Identity':
        return True, {}
    result = common_caffe_fields(node)
    supported = False
    name = None

    layer_type = result['type'].lower()
    if layer_type in lowered_keys_map:
        layer_type = lowered_keys_map[layer_type]
        assert layer_type in caffe_type_extractors
        name = layer_type

    if name:  # it is either standard or registered via CustomLayersMapping.xml
        attrs = caffe_type_extractors[name](node)
        # intentionally as Python registry if not found returns None
        if attrs is not None:
            result.update(attrs)
            supported = True

    if not supported:
        raise Error('Found custom layer "{}". Model Optimizer does not support this layer. '.format(node.id) +
                    'Please, implement extension. ' +
                    refer_to_faq_msg(45))

    if 'infer' not in result or not result['infer']:
        result.update(native_caffe_node_extractor(node))

    phase_attr = check_phase(node)
    result.update(phase_attr)
    return supported, result


def check_phase(node: Node):
    if node.has_valid('pb') and hasattr(node.pb, 'include'):
        for i in node.pb.include:
            if hasattr(i, 'phase'):
                return {'phase': i.phase}
    return {}


def register_caffe_python_extractor(op: Op, name: str = None):
    if not name and hasattr(op, 'op'):
        name = op.op
    if not name:
        raise Error("Can not register Op {}. Please, call function 'register_caffe_python_extractor' "
                    "with parameter 'name' .".format(op),
                    refer_to_faq_msg(87))
    CaffePythonFrontExtractorOp.registered_ops[name] = lambda node: extension_op_extractor(node, op)
