"""
 Copyright (C) 2018-2021 Intel Corporation

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

import numpy as np

from extensions.ops.mvn import MVN
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from extensions.ops.BatchNormInference import BatchNormInference
from extensions.ops.BatchNormInferenceMultipleOutputs import BatchNormInferenceMO

batchNormAttrList = ['data_format', 'data_type', 'eps', 'fix_gamma', 'shape', 'value']


class FusedBatchNormTraining(MiddleReplacementPattern):
    """
    Transformation looks for the BatchNorm layers in training mode and does the following:
    1. Fuses batch dimension with one of the spatial dimensions of the input to BatchNorm because batch normalization is
    performed over batch dimension also (per channel(features) dimension).
    2. Inserts MVN layer.
    3. Reshape MVN output back to the original one.
    """
    enabled = True
    replacement_id = "Fused_Batch_Norm_is_training_true"
    force_shape_inference = True
    force_clean_up = True
    # transformation works for the NHWC layout because transformation inserts Reshape to fuse N and H dimensions
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(kind='op', op='BatchNormTraining'))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        bn_train_node = match['op']
        bn_train_node_name = bn_train_node.name
        additional_attrs = {}
        for batchNormAttr in batchNormAttrList:
            if bn_train_node.has(batchNormAttr):
                additional_attrs[batchNormAttr] = bn_train_node[batchNormAttr]

        if len(bn_train_node.out_nodes().items()) > 1:
            additional_attrs['name'] = bn_train_node_name + '/batchNormInferenceMO'
            node = BatchNormInferenceMO(bn_train_node.graph, additional_attrs).create_node()
            for port_id, out_node in bn_train_node.out_nodes().items():
                bn_train_node.out_port(port_id).get_connection().set_source(node.out_port(port_id))

        elif len(bn_train_node.out_nodes().items()) == 1:
            additional_attrs['name'] = bn_train_node_name + '/batchNormInference'
            node = BatchNormInference(bn_train_node.graph, additional_attrs).create_node()
            bn_train_node.out_port(0).get_connection().set_source(node.out_port(0))

        else:
            assert False, 'Node  {} has not output nodes'.format(bn_train_node.name)

        for port_id, _ in bn_train_node.in_nodes().items():
            bn_train_node.in_port(port_id).get_connection().set_destination(node.in_port(port_id))

        shape = node.in_port(1).data.get_shape()
        assert shape is not None, 'The shape of scale input of the BatchNorm node {} is not defined'.format(node.name)

        bn_mean = Const(graph, {'name': node.name + '/mean', 'value': np.zeros(shape, dtype=np.float32),
                                'override_output_shape': True}).create_node()
        bn_std = Const(graph, {'name': node.name + '/std', 'value': np.ones(shape, dtype=np.float32),
                               'override_output_shape': True}).create_node()
        node.in_port(3).get_connection().set_source(bn_mean.out_port(0))
        node.in_port(4).get_connection().set_source(bn_std.out_port(0))

        # save the original shape
        original_shape = Shape(graph, {'name': node.in_port(0).get_source().node.soft_get('name')}).create_node()
        original_shape.in_port(0).connect(node.in_port(0).get_source())

        mvn = MVN(graph, {'name': node.name + '/mvn_', 'eps': node.soft_get('eps', 1e-6),
                          'override_output_shape': True}).create_node()
        node.in_port(0).get_connection().insert_node(mvn)

        reshape_4d = create_op_node_with_second_input(graph, Reshape, int64_array([1, -1, 0, 0]),
                                                      {'override_output_shape': True,
                                                       'name': node.soft_get('name') + '/fused_batch_and_channels'})
        mvn.in_port(0).get_connection().insert_node(reshape_4d)

        # restore original shape
        reshape_back = Reshape(graph, {'name': mvn.soft_get('name') + '/restore_shape',
                                       'override_output_shape': True}).create_node()
        reshape_back.in_port(1).connect(original_shape.out_port(0))
        mvn.out_port(0).get_connection().insert_node(reshape_back)
