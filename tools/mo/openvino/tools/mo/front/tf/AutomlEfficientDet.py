# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.Pack import Pack
from openvino.tools.mo.front.TransposeOrderNormalizer import TransposeOrderNormalizer
from openvino.tools.mo.front.eltwise_n import EltwiseNReplacement
from openvino.tools.mo.front.tf.pad_tf_to_pad import PadTFToPad
from openvino.tools.mo.ops.DetectionOutput import DetectionOutput
from openvino.tools.mo.ops.activation_ops import Sigmoid
from openvino.tools.mo.ops.priorbox_clustered import PriorBoxClusteredOp
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mo_array
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.result import Result


class EfficientDet(FrontReplacementFromConfigFileGeneral):
    replacement_id = 'AutomlEfficientDet'
    run_not_recursively = True

    def run_before(self):
        from openvino.tools.mo.front.ExpandDimsToUnsqueeze import ExpandDimsToUnsqueeze
        return [ExpandDimsToUnsqueeze, Pack, TransposeOrderNormalizer, PadTFToPad, EltwiseNReplacement]

    class AnchorGenerator:
        def __init__(self, min_level, aspect_ratios, num_scales, anchor_scale):
            self.min_level = min_level
            self.aspect_ratios = aspect_ratios
            self.anchor_scale = anchor_scale
            self.scales = [2 ** (float(s) / num_scales) for s in range(num_scales)]

        def get(self, layer_id):
            widths = []
            heights = []
            for s in self.scales:
                for a in self.aspect_ratios:
                    base_anchor_size = 2 ** (self.min_level + layer_id) * self.anchor_scale
                    heights.append(base_anchor_size * s * a[1])
                    widths.append(base_anchor_size * s * a[0])
            return widths, heights

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        parameter_node = graph.get_op_nodes(op='Parameter')[0]
        parameter_node['data_type'] = data_type_str_to_np(parameter_node.graph.graph['cmd_params'].data_type)

        # remove existing Result operations to remove unsupported sub-graph
        graph.remove_nodes_from([node.id for node in graph.get_op_nodes(op='Result')] + ['detections'])

        # determine if the op which is a input/final result of mean value and scale applying to the input tensor
        # then connect it to the input of the first convolution of the model, so we remove the image pre-processing
        # which includes padding and resizing from the model
        preprocessing_input_node_id = replacement_descriptions['preprocessing_input_node']
        assert preprocessing_input_node_id in graph.nodes, 'The node with name "{}" is not found in the graph. This ' \
                                                           'should be a last node before image normalization and is specified' \
                                                           ' in the json file.'.format(preprocessing_input_node_id)
        preprocessing_input_node = Node(graph, preprocessing_input_node_id)
        consumer_node = preprocessing_input_node.out_port(0).get_connection().get_destination().node
        consumer_node.in_port(0).get_connection().set_source(parameter_node.out_port(0))

        preprocessing_output_node_id = replacement_descriptions['preprocessing_output_node']
        assert preprocessing_output_node_id in graph.nodes, 'The node with name "{}" is not found in the graph. This ' \
                                                            'node should provide scaled image output and is specified' \
                                                            ' in the json file.'.format(preprocessing_output_node_id)
        preprocessing_output_node = Node(graph, preprocessing_output_node_id)
        preprocessing_output_node.out_port(0).disconnect()

        convolution_nodes = [n for n in graph.pseudo_topological_sort() if n.soft_get('type') == 'Convolution']
        convolution_nodes[0].in_port(0).get_connection().set_source(preprocessing_output_node.out_port(0))

        # create prior boxes (anchors) generator
        aspect_ratios = replacement_descriptions['aspect_ratios']
        assert len(aspect_ratios) % 2 == 0
        aspect_ratios = list(zip(aspect_ratios[::2], aspect_ratios[1::2]))
        priors_generator = self.AnchorGenerator(min_level=int(replacement_descriptions['min_level']),
                                                aspect_ratios=aspect_ratios,
                                                num_scales=int(replacement_descriptions['num_scales']),
                                                anchor_scale=replacement_descriptions['anchor_scale'])

        prior_boxes = []
        for i in range(100):
            inp_name = 'box_net/box-predict{}/BiasAdd'.format('_%d' % i if i else '')
            if inp_name not in graph:
                break
            widths, heights = priors_generator.get(i)
            prior_box_op = PriorBoxClusteredOp(graph, {'width': mo_array(widths),
                                                       'height': mo_array(heights),
                                                       'clip': 0, 'flip': 0,
                                                       'variance': replacement_descriptions['variance'],
                                                       'offset': 0.5})
            prior_boxes.append(prior_box_op.create_node([Node(graph, inp_name), parameter_node]))

        # concatenate prior box operations
        concat_prior_boxes = Concat(graph, {'axis': -1}).create_node()
        for idx, node in enumerate(prior_boxes):
            concat_prior_boxes.add_input_port(idx)
            concat_prior_boxes.in_port(idx).connect(node.out_port(0))

        conf = Sigmoid(graph, dict(name='concat/sigmoid')).create_node([Node(graph, 'concat')])
        reshape_size_node = Const(graph, {'value': int64_array([0, -1])}).create_node([])
        logits = Reshape(graph, dict(name=conf.name + '/Flatten')).create_node([conf, reshape_size_node])
        deltas = Reshape(graph, dict(name='concat_1/Flatten')).create_node([Node(graph, 'concat_1'), reshape_size_node])

        # revert convolution boxes prediction weights from yxYX to xyXY (convolutions share weights and bias)
        weights = Node(graph, 'box_net/box-predict/pointwise_kernel')
        weights.value = weights.value.reshape(-1, 4)[:, [1, 0, 3, 2]].reshape(weights.shape)
        bias = Node(graph, 'box_net/box-predict/bias')
        bias.value = bias.value.reshape(-1, 4)[:, [1, 0, 3, 2]].reshape(bias.shape)

        detection_output_node = DetectionOutput(graph, dict(
            name='detections',
            share_location=1,
            background_label_id=int(replacement_descriptions['num_classes']) + 1,
            nms_threshold=replacement_descriptions['nms_threshold'],
            confidence_threshold=replacement_descriptions['confidence_threshold'],
            top_k=100,
            keep_top_k=100,
            code_type='caffe.PriorBoxParameter.CENTER_SIZE',
        )).create_node([deltas, logits, concat_prior_boxes])

        output_op = Result(graph, dict(name='output'))
        output_op.create_node([detection_output_node])
