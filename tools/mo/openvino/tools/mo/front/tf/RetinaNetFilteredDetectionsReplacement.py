# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.DetectionOutput import DetectionOutput
from openvino.tools.mo.ops.elementwise import Mul, Sub, Pow
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.split import VariadicSplit
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array, mo_array
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.graph import clear_tensor_names_info


class RetinaNetFilteredDetectionsReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The class replaces the sub-graph that performs boxes post-processing and NMS with the DetectionOutput layer.

    The post-processing in the RetinaNet topology is performed differently from the DetectionOutput layer implementation
    in the OpenVINO. The first one calculates (d_x1, d_y1, d_x2, d_y2) which are a factor of the prior box width
    and height. The DetectionOuput with "code_type" equal to "caffe.PriorBoxParameter.CORNER" just adds predicted deltas
    to the prior box coordinates. This replacer add nodes which calculate prior box widths and heights, apply variances
    to the predicated box coordinates and multiply them. With this approach the DetectionOutput layer with "code_type"
    equal to "caffe.PriorBoxParameter.CORNER" produces the same result as the post-processing in the original topology.
    """
    replacement_id = 'RetinaNetFilteredDetectionsReplacement'

    def output_edges_match(self, graph: Graph, match: SubgraphMatch, new_sub_graph: dict):
        return {match.output_node(0)[0].id: new_sub_graph['detection_output_node'].id}

    def nodes_to_remove(self, graph: Graph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        new_nodes_to_remove.remove(match.single_input_node(0)[0].id)
        new_nodes_to_remove.remove(match.single_input_node(1)[0].id)
        new_nodes_to_remove.remove(match.single_input_node(2)[0].id)
        return new_nodes_to_remove

    @staticmethod
    def append_variances(priors_scale_node: Node, variance: list):
        graph = priors_scale_node.graph
        name = priors_scale_node.name

        sp_shape = Shape(graph, {'name': name + '/shape'}).create_node()
        priors_scale_node.out_port(0).connect(sp_shape.in_port(0))

        begin = Const(graph, {'value': int64_array([-2])}).create_node()
        end = Const(graph, {'value': int64_array([-1])}).create_node()
        stride = Const(graph, {'value': int64_array([1])}).create_node()
        shape_part_for_tiling = StridedSlice(graph, {'name': name + '/get_-2_dim', 'begin_mask': int64_array([1]),
                                                     'end_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
                                                     'shrink_axis_mask': int64_array([0]),
                                                     'ellipsis_mask': int64_array([0])}).create_node()

        sp_shape.out_port(0).connect(shape_part_for_tiling.in_port(0))
        begin.out_port(0).connect(shape_part_for_tiling.in_port(1))
        end.out_port(0).connect(shape_part_for_tiling.in_port(2))
        stride.out_port(0).connect(shape_part_for_tiling.in_port(3))

        shape_concat = create_op_node_with_second_input(graph, Concat, int64_array([4]),
                                                        {'name': name + '/shape_for_tiling', 'in_ports_count': 2,
                                                         'axis': int64_array(0)},
                                                        shape_part_for_tiling)

        variance = Const(graph, {'name': name + '/variance', 'value': float32_array(variance)}).create_node()
        tile = Broadcast(graph, {'name': name + '/variance_tile'}).create_node()
        variance.out_port(0).connect(tile.in_port(0))
        shape_concat.out_port(0).connect(tile.in_port(1))

        reshape_dim = Const(graph, {'value': int64_array([-1, 4])}).create_node()
        sp_reshape = Reshape(graph, {'name': name + '/reshape'}).create_node()
        sp_reshape.in_port(0).connect(priors_scale_node.out_port(0))
        sp_reshape.in_port(1).connect(reshape_dim.out_port(0))

        concat = Concat(graph,
                        {'name': name + '/priors_concat', 'axis': int64_array(0), 'in_ports_count': 2}).create_node()
        sp_reshape.out_port(0).connect(concat.in_port(0))
        tile.out_port(0).connect(concat.in_port(1))

        output_dims = Const(graph, {'value': int64_array([1, 2, -1])}).create_node()
        output_node = Reshape(graph, {'name': name + '/3D_priors_wth_variances'}).create_node()
        concat.out_port(0).connect(output_node.in_port(0))
        output_dims.out_port(0).connect(output_node.in_port(1))

        return output_node

    def placeholder_scales(self, placeholder: Node):
        """
        Helper function to get scales for prior boxes out of input image size:
                [1 / im_width, 1 / im_height, 1 / im_width, 1 / im_height]
        """
        graph = placeholder.graph
        name = placeholder.soft_get('name', placeholder.id)

        shape_value = placeholder.soft_get('shape', None)
        assert shape_value is not None, \
            "[ {} replacer ] Placeholder `{}` should have shape attribute".format(self.replacement_id, name)
        assert isinstance(shape_value, np.ndarray), \
            "[ {} replacer ] Placeholder `{}` shape attribute should be np.ndarray".format(self.replacement_id, name)
        assert shape_value.size == 4, \
            "[ {} replacer ] Placeholder `{}` should be 4D. Shape: {}".format(self.replacement_id, name, shape_value)

        shape = Shape(graph, {'name': 'input_image_shape'}).create_node()
        shape.in_port(0).connect(placeholder.out_port(0))

        begin = Const(graph, {'value': int64_array([1])}).create_node()
        end = Const(graph, {'value': int64_array([3])}).create_node()
        stride = Const(graph, {'value': int64_array([1])}).create_node()
        spatial = StridedSlice(graph, {'name': name + '/get_h_w', 'begin_mask': int64_array([1]),
                                       'end_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
                                       'shrink_axis_mask': int64_array([0]), 'ellipsis_mask': int64_array([0])}).create_node()

        spatial.in_port(0).connect(shape.out_port(0))
        spatial.in_port(1).connect(begin.out_port(0))
        spatial.in_port(2).connect(end.out_port(0))
        spatial.in_port(3).connect(stride.out_port(0))

        power = Const(graph, {'value': float32_array([-1.])}).create_node()
        spatial_scale = Pow(graph, {}).create_node()

        spatial_scale.in_port(0).connect(spatial.out_port(0))
        spatial_scale.in_port(1).connect(power.out_port(0))

        # Power `type_infer` requires inputs to have equal data type
        convert_to_fp32 = Cast(graph, {'dst_type': np.float32}).create_node()
        spatial_scale.in_port(0).get_connection().insert_node(convert_to_fp32)

        order = Const(graph, {'value': int64_array([1, 0])}).create_node()
        axis_const = Const(graph, {'value': int64_array(0)}).create_node()
        reverse = Gather(graph, {}).create_node()

        reverse.in_port(0).connect(spatial_scale.out_port(0))
        reverse.in_port(1).connect(order.out_port(0))
        axis_const.out_port(0).connect(reverse.in_port(2))

        priors_scale_node = Concat(graph, {'axis': 0, 'in_ports_count': 2}).create_node()
        priors_scale_node.add_input_port(0, skip_if_exist=True)
        priors_scale_node.add_input_port(1, skip_if_exist=True)

        priors_scale_node.in_port(0).connect(reverse.out_port(0))
        priors_scale_node.in_port(1).connect(reverse.out_port(0))
        return priors_scale_node

    def generate_sub_graph(self, graph: Graph, match: SubgraphMatch):
        reshape_classes_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                                dict(name='do_reshape_classes'),
                                                                match.single_input_node(1)[0])

        initial_priors_node = match.single_input_node(2)[0]
        priors_name = initial_priors_node.soft_get('name', initial_priors_node.id)
        # model calculates identical prior boxes for each batch, so we take first slice of them
        begin = Const(graph, {'value': mo_array([0, 0, 0], dtype=np.int32)}).create_node()
        end = Const(graph, {'value': mo_array([1, 0, 0], dtype=np.int32)}).create_node()
        stride = Const(graph, {'value': mo_array([1, 1, 1], dtype=np.int32)}).create_node()

        priors_node = StridedSlice(graph, {'name': priors_name + '/0_batch_slice',
                                           'begin_mask': int64_array([1, 1, 1]),
                                           'end_mask': int64_array([1, 0, 0]),
                                           'new_axis_mask': int64_array([0]),
                                           'shrink_axis_mask': int64_array([0]),
                                           'ellipsis_mask': int64_array([0])}).create_node()

        initial_priors_node.out_port(0).connect(priors_node.in_port(0))
        begin.out_port(0).connect(priors_node.in_port(1))
        end.out_port(0).connect(priors_node.in_port(2))
        stride.out_port(0).connect(priors_node.in_port(3))

        placeholders = graph.get_op_nodes(type='Parameter')
        assert len(placeholders) == 1, "{} replacer requires model to have one Placeholder, but current model has " \
                                       "{} placeholders".format(self.replacement_id, len(placeholders))
        placeholder = placeholders[0]

        # scale prior boxes to the [0, 1] interval
        node_with_scales_for_prior_boxes = self.placeholder_scales(placeholder)
        priors_scale_node = Mul(graph, {'name': 'scale_priors'}).create_node()

        broadcast = Broadcast(graph, {'name': 'scales_broadcast'}).create_node()
        shape_of_priors = Shape(graph, {'name': 'priors_shape'}).create_node()
        priors_node.out_port(0).connect(shape_of_priors.in_port(0))
        broadcast.in_port(1).connect(shape_of_priors.out_port(0))
        broadcast.in_port(0).connect(node_with_scales_for_prior_boxes.out_port(0))

        priors_scale_node.in_port(0).connect(priors_node.out_port(0))
        priors_scale_node.in_port(1).connect(broadcast.out_port(0))

        try:
            variance = match.custom_replacement_desc.custom_attributes['variance']
        except:
            raise Error('There is no variance attribute in {} replacement config file `custom_attributes`'
                        ''.format(self.replacement_id))

        priors = self.append_variances(priors_scale_node, variance)

        # calculate prior boxes widths and heights
        split_node = create_op_with_const_inputs(
            graph, VariadicSplit, {1: int64_array(2), 2: int64_array([1, 1, 1, 1])}, {'out_ports_count': 4},
            priors_scale_node)

        priors_width_node = Sub(graph, dict(name=split_node.name + '/sub_2-0_')
                                ).create_node([(split_node, 2), (split_node, 0)])
        priors_height_node = Sub(graph, dict(name=split_node.name + '/sub_3-1_')
                                 ).create_node([(split_node, 3), (split_node, 1)])

        # concat weights and heights into a single tensor and multiple with the box coordinates regression values
        # WA with 3 Concats instead of 1 for keeping model reshapable
        # concat_width_height_node = Concat(graph, {'name': 'concat_priors_width_height', 'axis': -1,
        #                                           'in_ports_count': 4}).create_node(
        # [priors_width_node, priors_height_node, priors_width_node, priors_height_node])

        concat_1 = Concat(graph, {'name': 'concat_width_height',
                                  'axis': -1, 'in_ports_count': 2}).create_node([priors_width_node, priors_height_node])
        concat_2 = Concat(graph, {'name': 'concat_width_height_width',
                                  'axis': -1, 'in_ports_count': 2}).create_node([concat_1, priors_width_node])
        concat_width_height_node = Concat(graph, {'name': 'concat_priors_width_height', 'axis': -1, 'in_ports_count': 2}
                                          ).create_node([concat_2, priors_height_node])

        applied_width_height_regressions_node = Mul(graph, {'name': 'final_regressions'}).create_node(
            [concat_width_height_node, match.single_input_node(0)[0]])

        # reshape to 2D tensor as OpenVINO Detection Output layer expects
        reshape_regression_node = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                                   dict(name='reshape_regression'),
                                                                   applied_width_height_regressions_node)

        detection_output_op = DetectionOutput(graph, match.custom_replacement_desc.custom_attributes)
        # get nms from the original network
        iou_threshold = None
        nms_nodes = graph.get_op_nodes(op='NonMaxSuppression')
        if len(nms_nodes) > 0:
            # it is highly unlikely that for different classes NMS has different
            # moreover DetectionOutput accepts only scalar values for iou_threshold (nms_threshold)
            iou_threshold = nms_nodes[0].in_node(3).value
        if iou_threshold is None:
            raise Error('During {} `iou_threshold` was not retrieved from RetinaNet graph'.format(self.replacement_id))

        detection_output_node = detection_output_op.create_node(
            [reshape_regression_node, reshape_classes_node, priors],
            dict(name=detection_output_op.attrs['type'], nms_threshold=iou_threshold, clip_after_nms=1, normalized=1,
                 variance_encoded_in_target=0, background_label_id=1000))

        # As outputs are replaced with a postprocessing node, outgoing tensor names are no longer
        # correspond to original tensors and should be removed from output->Result edges
        out_nodes = []
        for out in range(match.outputs_count()):
            out_nodes.append(match.output_node(out)[0])
        clear_tensor_names_info(out_nodes)

        return {'detection_output_node': detection_output_node}
