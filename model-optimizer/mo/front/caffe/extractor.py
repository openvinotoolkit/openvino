"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.caffe.extractors.batchnorm import batch_norm_ext
from mo.front.caffe.extractors.concat import concat_ext
from mo.front.caffe.extractors.inner_product import inner_product_ext
from mo.front.caffe.extractors.lrn import lrn_ext
from mo.front.caffe.extractors.native_caffe import native_caffe_node_extractor
from mo.front.caffe.extractors.reshape import reshape_ext
from mo.front.caffe.extractors.roipooling import roipooling_ext
from mo.front.caffe.extractors.scale import scale_ext
from mo.front.caffe.extractors.slice import slice_ext
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.register_custom_ops import extension_op_extractor
from mo.front.extractor import CaffePythonFrontExtractorOp
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def node_pb_arg(pb_extractor):
    return lambda node: pb_extractor(node.pb, node.model_pb)


"""
Keys are names that appear as layer names in .prototxt.
Full list is available here: http://caffe.berkeleyvision.org/tutorial/layers.html
"""
caffe_type_extractors = {
    # Common Layers
    'innerproduct': node_pb_arg(inner_product_ext),
    'inner_product': node_pb_arg(inner_product_ext),
    'dropout': node_pb_arg(lambda _, __: dict(op='Dropout', infer=copy_shape_infer)),

    # Normalization Layers
    'batchnorm': node_pb_arg(batch_norm_ext),
    'lrn': node_pb_arg(lrn_ext),

    # Activation Layers
    'scale': node_pb_arg(scale_ext),

    # Utility Layers
    'concat': node_pb_arg(concat_ext),
    'reshape': node_pb_arg(reshape_ext),
    'slice': node_pb_arg(slice_ext),

    # Custom, implemented in IE, Fast-RCNN-specific
    'roipooling': node_pb_arg(roipooling_ext),
}


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
        'precision': 'FP32'  # TODO use real precision derived from the model
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
        raise Error("Can not register Op {}. Please, call function 'register_caffe_python_extractor'"
                    "with parameter 'name' .".format(op),
                    refer_to_faq_msg(87))
    CaffePythonFrontExtractorOp.registered_ops[name] = lambda node: extension_op_extractor(node, op)
