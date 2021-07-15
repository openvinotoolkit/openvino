# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from mo.ops.reshape import Reshape
from mo.ops.const import Const
from mo.ops.concat import Concat
from extensions.ops.transpose import Transpose
from extensions.ops.split import VariadicSplit
from extensions.ops.elementwise import Less, Mul, Greater, Sub, Add, Div
from extensions.ops.select import Select
from extensions.ops.activation_ops import Sigmoid, Exp

from extensions.front.no_op_eraser import NoOpEraser
from extensions.ops.regionyolo import RegionYoloOp
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Node, Graph
from mo.ops.result import Result
from mo.utils.error import Error


class YoloRegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all Result nodes in graph with YoloRegion->Result nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLO'

    def run_after(self):
        return [NoOpEraser]

    def transform_graph(self, graph: Graph, replacement_descriptions):
        op_outputs = [n for n, d in graph.nodes(data=True) if 'op' in d and d['op'] == 'Result']
        for op_output in op_outputs:
            last_node = Node(graph, op_output).in_node(0)
            op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1)
            op_params.update(replacement_descriptions)
            region_layer = RegionYoloOp(graph, op_params)
            region_layer_node = region_layer.create_node([last_node])
            # here we remove 'axis' from 'dim_attrs' to avoid permutation from axis = 1 to axis = 2
            region_layer_node.dim_attrs.remove('axis')
            Result(graph).create_node([region_layer_node])
            graph.remove_node(op_output)


class YoloV3RegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all Result nodes in graph with YoloRegion->Result nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLOV3'


    def transform_graph(self, graph: Graph, replacement_descriptions):
        include_postprocess = replacement_descriptions.get('include_postprocess', False)
        if include_postprocess:
            input_node = Node(graph, graph.get_nodes_with_attributes(op='Parameter')[0])
            hNorm = input_node.shape[1]
            wNorm = input_node.shape[2]
            hw_norm = 32

        graph.remove_nodes_from(graph.get_nodes_with_attributes(op='Result'))
        for i, input_node_name in enumerate(replacement_descriptions['entry_points']):
            if input_node_name not in graph.nodes():
                raise Error('TensorFlow YOLO V3 conversion mechanism was enabled. '
                            'Entry points "{}" were provided in the configuration file. '
                            'Entry points are nodes that feed YOLO Region layers. '
                            'Node with name {} doesn\'t exist in the graph. '
                            'Refer to documentation about converting YOLO models for more information.'.format(
                    ', '.join(replacement_descriptions['entry_points']), input_node_name))
            last_node = Node(graph, input_node_name).in_node(0)
            anchors = replacement_descriptions['anchors']

            op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1, do_softmax=0)
            op_params.update(replacement_descriptions)
            if 'masks' in op_params:
                op_params['mask'] = op_params['masks'][i]
                anchors = anchors[op_params['mask'][0] * 2 : op_params['mask'][-1] * 2 + 2]
                del op_params['masks']
            region_layer_node = RegionYoloOp(graph, op_params).create_node([last_node])
            # TODO: do we need change axis for further permutation
            region_layer_node.dim_attrs.remove('axis')

            if not include_postprocess:
                Result(graph, {'name': region_layer_node.id + '/Result'}).create_node([region_layer_node])
                continue

            rows = replacement_descriptions['classes'] + replacement_descriptions['coords'] + 1

            shape_node = Const(graph, {'value': np.array([-1, rows], dtype=np.int)}).create_node()
            tr_axes = Const(graph, {'value': np.array([1, 0], dtype=np.int)}).create_node()

            input2d = Reshape(graph, {'special_zero': True}).create_node([last_node, shape_node])
            input2d = Transpose(graph, {}).create_node([input2d, tr_axes])

            # Do not remove! This layer preserves NHWC consistency.
            tr_axes_3d =  Const(graph, {'value': np.array([0, 1, 2, 3], dtype=np.int)}).create_node()
            region_layer_node = Transpose(graph, {}).create_node([region_layer_node, tr_axes_3d])

            shape_node = Const(graph, {'value': np.array([-1, rows], dtype=np.int)}).create_node()
            region_layer_node = Reshape(graph, {'special_zero': True}).create_node([region_layer_node, shape_node])
            region_layer_node = Transpose(graph, {}).create_node([region_layer_node, tr_axes])

            axes = Const(graph, {'value': 0}).create_node()
            splits = Const(graph, {'value': np.array([1, 1, 1, 1, rows - 4], dtype=np.int)}).create_node()
            split = VariadicSplit(graph, {'out_ports_count': 5}).create_node([input2d, axes, splits])

            h = int(hNorm / hw_norm)
            w = int(wNorm / hw_norm)
            hw_norm /= 2
            anchors_num = int(len(anchors) / 2)

            shift_node = Const(graph, {'value': np.array([0.5], dtype=np.float)}).create_node()
            shape_3d = Const(graph, {'value': np.array([-1, anchors_num, h, w], dtype=np.int)}).create_node()
            box_broad_shape = (1, anchors_num, h, w)

            box_x = Sigmoid(graph, {}).create_node()
            box_x.in_port(0).get_connection().set_source(split.out_port(0))

            box_x = Sub(graph, {}).create_node([box_x, shift_node])
            box_x = Add(graph, {}).create_node([box_x, shift_node])
            box_x = Reshape(graph, {'special_zero': True}).create_node([box_x, shape_3d])

            x_inices = np.zeros(w * h * anchors_num, dtype=np.float)
            for k in range(h):
                x_inices[k * anchors_num: (k + 1) * anchors_num] = k

            for j in range(1, w):
                step = h * anchors_num
                x_inices[j * step: j * step + step] = x_inices[:step]

            x_inices = np.reshape(x_inices, box_broad_shape)
            horiz = Const(graph, {'value': x_inices}).create_node()
            box_x = Add(graph, {'name': 'Add_'}).create_node([box_x, horiz])

            cols_node = Const(graph, {'value': float(w)}).create_node()
            box_x = Div(graph, {}).create_node([box_x, cols_node])

            box_y = Sigmoid(graph, {}).create_node()
            box_y.in_port(0).get_connection().set_source(split.out_port(1))
            box_y = Sub(graph, {}).create_node([box_y, shift_node])
            box_y = Add(graph, {}).create_node([box_y, shift_node])
            box_y = Reshape(graph, {'special_zero': True}).create_node([box_y, shape_3d])

            y_inices = np.zeros(h * anchors_num, dtype=np.float)
            for k in range(h):
                y_inices[k * anchors_num: (k + 1) * anchors_num] = k

            y_inices = np.reshape(y_inices, (1, anchors_num, h, 1))
            vert =  Const(graph, {'value': y_inices}).create_node()
            box_y = Add(graph, {}).create_node([box_y, vert])

            rows_node = Const(graph, {'value': float(h)}).create_node()
            box_y = Div(graph, {}).create_node([box_y, rows_node])

            anchors_w = np.zeros(anchors_num, dtype=np.float)
            anchors_h = np.zeros(anchors_num, dtype=np.float)

            for k in range(anchors_num):
                anchors_w[k] = anchors[2 * k] / wNorm
                anchors_h[k] = anchors[2 * k + 1] / hNorm

            bias_w = np.zeros(w * h * anchors_num, dtype=np.float)
            bias_h = np.zeros(w * h * anchors_num, dtype=np.float)
            for k in range(h):
                bias_w[k * anchors_num: (k+1) * anchors_num] = anchors_w
                bias_h[k * anchors_num: (k+1) * anchors_num] = anchors_h

            for k in range(w):
                step = h * anchors_num
                bias_w[k * step:(k+1) * step] = bias_w[:step]
                bias_h[k * step:(k+1) * step] = bias_h[:step]

            bias_w = np.reshape(bias_w, box_broad_shape)
            bias_h = np.reshape(bias_h, box_broad_shape)

            box_w = Exp(graph, {}).create_node()
            box_w.in_port(0).get_connection().set_source(split.out_port(2))
            box_w = Reshape(graph, {'special_zero': True}).create_node([box_w, shape_3d])

            anchor_w_node = Const(graph, {'value': bias_w}).create_node()
            box_w = Mul(graph, {}).create_node([box_w, anchor_w_node])

            box_h = Exp(graph, {}).create_node()
            box_h.in_port(0).get_connection().set_source(split.out_port(3))
            box_h = Reshape(graph, {'special_zero': True}).create_node([box_h, shape_3d])
            anchor_h_node = Const(graph, {'value': bias_h}).create_node()
            box_h = Mul(graph, {}).create_node([box_h, anchor_h_node])

            rs_axis = Const(graph, {'value': 0}).create_node()
            rs_splits = Const(graph, {'value': np.array([4, 1, rows - 5], dtype=np.int)}).create_node()
            region_split = VariadicSplit(graph, {'out_ports_count': 3}).create_node([region_layer_node, rs_axis, rs_splits])

            probs = Mul(graph, {}).create_node()
            probs.in_port(0).get_connection().set_source(region_split.out_port(2))
            probs.in_port(1).get_connection().set_source(region_split.out_port(1))

            concat_shape = Const(graph, {'value': np.array([1, -1], dtype=np.float)}).create_node()
            box_x = Reshape(graph, {'special_zero': True}).create_node([box_x, concat_shape])
            box_y = Reshape(graph, {'special_zero': True}).create_node([box_y, concat_shape])
            box_w = Reshape(graph, {'special_zero': True}).create_node([box_w, concat_shape])
            box_h = Reshape(graph, {'special_zero': True}).create_node([box_h, concat_shape])

            result = Concat(graph, dict(in_ports_count=6, axis=0)).create_node()
            result.in_port(0).connect(box_x.out_port(0))
            result.in_port(1).connect(box_y.out_port(0))
            result.in_port(2).connect(box_w.out_port(0))
            result.in_port(3).connect(box_h.out_port(0))
            result.in_port(4).connect(region_split.out_port(1))
            result.in_port(5).connect(probs.out_port(0))

            result = Transpose(graph, dict(name = result.id + '/ResWithPostprocess')).create_node([result, tr_axes])

            Result(graph, {'name': result.id + '/Result'}).create_node([result])


class YoloV4RegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all Result nodes in graph with YoloRegion->Result nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLOV4'

    def run_after(self):
        return [NoOpEraser]

    def transform_graph(self, graph: Graph, replacement_descriptions):
        # print(graph.nodes())
        include_postprocess = replacement_descriptions.get('include_postprocess', False)
        if include_postprocess:
            input_node = Node(graph, graph.get_nodes_with_attributes(op='Parameter')[0])
            hNorm = input_node.shape[1]
            wNorm = input_node.shape[2]
            hw_norm = 32
        else:
            return

        graph.remove_nodes_from(graph.get_nodes_with_attributes(op='Result'))
        for i, input_node_name in enumerate(replacement_descriptions['entry_points']):
            if input_node_name not in graph.nodes():
                raise Error('TensorFlow YOLO V3 conversion mechanism was enabled. '
                            'Entry points "{}" were provided in the configuration file. '
                            'Entry points are nodes that feed YOLO Region layers. '
                            'Node with name {} doesn\'t exist in the graph. '
                            'Refer to documentation about converting YOLO models for more information.'.format(
                    ', '.join(replacement_descriptions['entry_points']), input_node_name))
            last_node = Node(graph, input_node_name).in_node(0)
            anchors = replacement_descriptions['anchors']

            op_params = dict()
            op_params.update(replacement_descriptions)
            if 'masks' in op_params:
                op_params['mask'] = op_params['masks'][i]
                anchors = anchors[op_params['mask'][0] * 2 : op_params['mask'][-1] * 2 + 2]
                del op_params['masks']

            rows = replacement_descriptions['classes'] + replacement_descriptions['coords'] + 1

            shape_node = Const(graph, {'value': np.array([-1, rows], dtype=np.int)}).create_node()
            tr_axes = Const(graph, {'value': np.array([1, 0], dtype=np.int)}).create_node()

            input2d = Reshape(graph, {'special_zero': True}).create_node([last_node, shape_node])
            input2d = Transpose(graph, {}).create_node([input2d, tr_axes])

            axes = Const(graph, {'value': 0}).create_node()
            splits = Const(graph, {'value': np.array([1, 1, 1, 1, rows - 4], dtype=np.int)}).create_node()
            split = VariadicSplit(graph, {'out_ports_count': 5}).create_node([input2d, axes, splits])

            h = int(hNorm / hw_norm)
            w = int(wNorm / hw_norm)
            hw_norm /= 2
            anchors_num = int(len(anchors) / 2)

            shape_3d = Const(graph, {'value': np.array([-1, anchors_num, h, w], dtype=np.int)}).create_node()
            box_broad_shape = (1, anchors_num, h, w)

            box_x = Sigmoid(graph, {}).create_node()
            box_x.in_port(0).get_connection().set_source(split.out_port(0))
            box_x = Reshape(graph, {'special_zero': True}).create_node([box_x, shape_3d])

            x_inices = np.zeros(w * h * anchors_num, dtype=np.float)
            for k in range(h):
                x_inices[k * anchors_num: (k + 1) * anchors_num] = k

            for j in range(1, w):
                step = h * anchors_num
                x_inices[j * step: j * step + step] = x_inices[:step]

            x_inices = np.reshape(x_inices, box_broad_shape)
            horiz = Const(graph, {'value': x_inices}).create_node()
            box_x = Add(graph, {'name': 'Add_'}).create_node([box_x, horiz])

            cols_node = Const(graph, {'value': float(w)}).create_node()
            box_x = Div(graph, {}).create_node([box_x, cols_node])

            box_y = Sigmoid(graph, {}).create_node()
            box_y.in_port(0).get_connection().set_source(split.out_port(1))
            box_y = Reshape(graph, {'special_zero': True}).create_node([box_y, shape_3d])

            y_inices = np.zeros(h * anchors_num, dtype=np.float)
            for k in range(h):
                y_inices[k * anchors_num: (k + 1) * anchors_num] = k

            y_inices = np.reshape(y_inices, (1, anchors_num, h, 1))
            vert =  Const(graph, {'value': y_inices}).create_node()
            box_y = Add(graph, {}).create_node([box_y, vert])

            rows_node = Const(graph, {'value': float(h)}).create_node()
            box_y = Div(graph, {}).create_node([box_y, rows_node])

            anchors_w = np.zeros(anchors_num, dtype=np.float)
            anchors_h = np.zeros(anchors_num, dtype=np.float)

            for k in range(anchors_num):
                anchors_w[k] = anchors[2 * k] / wNorm
                anchors_h[k] = anchors[2 * k + 1] / hNorm

            bias_w = np.zeros(w * h * anchors_num, dtype=np.float)
            bias_h = np.zeros(w * h * anchors_num, dtype=np.float)
            for k in range(h):
                bias_w[k * anchors_num: (k+1) * anchors_num] = anchors_w
                bias_h[k * anchors_num: (k+1) * anchors_num] = anchors_h

            for k in range(w):
                step = h * anchors_num
                bias_w[k * step:(k+1) * step] = bias_w[:step]
                bias_h[k * step:(k+1) * step] = bias_h[:step]

            bias_w = np.reshape(bias_w, box_broad_shape)
            bias_h = np.reshape(bias_h, box_broad_shape)

            box_w = Exp(graph, {}).create_node()
            box_w.in_port(0).get_connection().set_source(split.out_port(2))
            box_w = Reshape(graph, {'special_zero': True}).create_node([box_w, shape_3d])

            anchor_w_node = Const(graph, {'value': bias_w}).create_node()
            box_w = Mul(graph, {}).create_node([box_w, anchor_w_node])

            box_h = Exp(graph, {}).create_node()
            box_h.in_port(0).get_connection().set_source(split.out_port(3))
            box_h = Reshape(graph, {'special_zero': True}).create_node([box_h, shape_3d])
            anchor_h_node = Const(graph, {'value': bias_h}).create_node()
            box_h = Mul(graph, {}).create_node([box_h, anchor_h_node])

            tr_axes_3d =  Const(graph, {'value': np.array([0, 1, 2, 3], dtype=np.int)}).create_node()
            region_layer_node = Transpose(graph, {}).create_node([last_node, tr_axes_3d])

            shape_node = Const(graph, {'value': np.array([-1, rows], dtype=np.int)}).create_node()
            region_layer_node = Reshape(graph, {'special_zero': True}).create_node([region_layer_node, shape_node])
            region_layer_node = Transpose(graph, {}).create_node([region_layer_node, tr_axes])

            rs_axis = Const(graph, {'value': 0}).create_node()
            rs_splits = Const(graph, {'value': np.array([4, 1, rows - 5], dtype=np.int)}).create_node()
            region_split = VariadicSplit(graph, {'out_ports_count': 3}).create_node([region_layer_node, rs_axis, rs_splits])

            object_probability = Sigmoid(graph, {}).create_node()
            object_probability.in_port(0).get_connection().set_source(region_split.out_port(1))

            probs_orig = Sigmoid(graph, {}).create_node()
            probs_orig.in_port(0).get_connection().set_source(region_split.out_port(2))

            probs = Mul(graph, {}).create_node()
            probs.in_port(0).get_connection().set_source(probs_orig.out_port(0))
            probs.in_port(1).get_connection().set_source(object_probability.out_port(0))

            concat_shape = Const(graph, {'value': np.array([1, -1], dtype=np.float)}).create_node()
            box_x = Reshape(graph, {'special_zero': True}).create_node([box_x, concat_shape])
            box_y = Reshape(graph, {'special_zero': True}).create_node([box_y, concat_shape])
            box_w = Reshape(graph, {'special_zero': True}).create_node([box_w, concat_shape])
            box_h = Reshape(graph, {'special_zero': True}).create_node([box_h, concat_shape])

            result = Concat(graph, dict(in_ports_count=6, axis=0)).create_node()
            result.in_port(0).connect(box_x.out_port(0))
            result.in_port(1).connect(box_y.out_port(0))
            result.in_port(2).connect(box_w.out_port(0))
            result.in_port(3).connect(box_h.out_port(0))
            result.in_port(4).connect(object_probability.out_port(0))
            result.in_port(5).connect(probs.out_port(0))

            result = Transpose(graph, dict(name = result.id + '/ResWithPostprocess')).create_node([result, tr_axes])
            Result(graph, {'name': result.id + '/Result'}).create_node([result])
