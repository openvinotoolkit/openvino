# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.op import Op


def get_attrs(node: Node):
    shapes = node.pb.attr["_output_shapes"].list.shape
    tf_types = node.pb.attr["component_types"].list.type
    extracted_types = []
    for t in tf_types:
        extracted_types.append(tf_dtype_extractor(t))
    result_shapes = []
    for shape_pb in shapes:
        result_shapes.append(tf_tensor_shape(shape_pb))
    assert len(result_shapes) == len(extracted_types), "Output shapes do not match output" \
                                                       "types in the node {}".format(node.soft_get('name', node.id))
    attrs = {"shapes": result_shapes, "types": extracted_types, 'out_ports_count': len(result_shapes)}
    return attrs


class QueueDequeueV1Extractor(FrontExtractorOp):
    op = "QueueDequeue"
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_attrs(node)
        Op.update_node_stat(node, attrs)
        return cls.enabled


class QueueDequeueV2Extractor(FrontExtractorOp):
    op = "QueueDequeueV2"
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_attrs(node)
        Op.update_node_stat(node, attrs)
        return cls.enabled
