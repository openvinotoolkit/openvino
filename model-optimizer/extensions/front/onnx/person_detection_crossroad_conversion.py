"""
 Copyright (c) 2018-2021 Intel Corporation

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

from extensions.ops.roifeatureextractor_onnx import ExperimentalDetectronROIFeatureExtractor
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Graph, Node, rename_node


class ONNXPersonDetectionCrossroadReplacement(FrontReplacementFromConfigFileGeneral):
    """
    Insert ExperimentalDetectronROIFeatureExtractor layers instead of sub-graphs of the model.
    """
    replacement_id = 'ONNXPersonDetectionCrossroadReplacement'

    def transform_graph(self, graph: Graph, replacement_descriptions: dict):
        fpn_heads = replacement_descriptions['fpn_heads']
        for inp, out in zip(replacement_descriptions['ROI_feature_extractor_inputs'],
                            replacement_descriptions['ROI_feature_extractor_outputs']):
            insert_experimental_layers(graph, fpn_heads, inp, out)


def insert_experimental_layers(graph: Graph, input_fpn_heads: list, inp: str, out: str):
    old_output_node = Node(graph, out)
    output_name = old_output_node.soft_get('name', old_output_node.id)
    old_output_node_name = output_name + '/old'
    rename_node(old_output_node, old_output_node_name)

    input_fpn_head_nodes = [Node(graph, node_id) for node_id in input_fpn_heads]
    fpn_roi_align = ExperimentalDetectronROIFeatureExtractor(graph, {'name': output_name,
                                                                     'output_size': 7,
                                                                     'pyramid_scales': int64_array(
                                                                         [4, 8, 16, 32, 64]),
                                                                     'sampling_ratio': 2, }).create_node()
    rename_node(fpn_roi_align, output_name)
    fpn_roi_align.in_port(0).connect(Node(graph, inp).out_port(0))
    for ind, fpn_node in enumerate(input_fpn_head_nodes):
        fpn_roi_align.in_port(ind + 1).connect(fpn_node.out_port(0))

    old_output_node.out_port(0).get_connection().set_source(fpn_roi_align.out_port(0))
