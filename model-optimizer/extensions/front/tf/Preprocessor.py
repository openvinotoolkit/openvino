"""
 Copyright (c) 2018 Intel Corporation

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

import logging as log
import networkx as nx

from extensions.front.sub import Sub
from extensions.front.tf.Pack import Pack
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph
from mo.graph.graph import create_edge, Node
from mo.utils.error import Error


class PreprocessorReplacement(FrontReplacementFromConfigFileSubGraph):
    """
    The class replaces the "Preprocessor" block resizing input image and applying mean/scale values. Only nodes related
    to applying mean/scaling values are kept.
    """
    replacement_id = 'PreprocessorReplacement'

    def run_before(self):
        return [Pack, Sub]

    def nodes_to_remove(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        new_nodes_to_remove = match.matched_nodes_names()
        # do not remove nodes that perform input image scaling and mean value subtraction
        for node_to_keep in ('Preprocessor/sub', 'Preprocessor/sub/y', 'Preprocessor/mul', 'Preprocessor/mul/x'):
            if node_to_keep in new_nodes_to_remove:
                new_nodes_to_remove.remove(node_to_keep)
        return new_nodes_to_remove

    def generate_sub_graph(self, graph: nx.MultiDiGraph, match: SubgraphMatch):
        print('WARNING: the "{}" is a legacy replacer that will be removed in the future release. Please, consider '
              'using replacers defined in the "extensions/front/tf/ObjectDetectionAPI.py"'.format(self.replacement_id))
        log.debug('PreprocessorReplacement: matched_nodes = {}'.format(match.matched_nodes_names()))

        sub_node = match.output_node(0)[0]
        if not sub_node.has('op') or sub_node.op != 'Sub':
            raise Error('The output op of the Preprocessor sub-graph is not of type "Sub". Looks like the topology is '
                        'not created with TensorFlow Object Detection API.')

        mul_node = None
        if sub_node.in_node(0).has('op') and sub_node.in_node(0).op == 'Mul':
            log.info('There is image scaling node in the Preprocessor block.')
            mul_node = sub_node.in_node(0)

        config_attrs = match.custom_replacement_desc.custom_attributes
        preprocessed_image_height_width = self.get_preprocessed_image_size_from_model(graph)
        if preprocessed_image_height_width is None:
            if 'preprocessed_image_width' not in config_attrs or 'preprocessed_image_height' not in config_attrs:
                raise Error('Failed to determine the pre-processed image size from the original TensorFlow graph. '
                            'Please, specify "preprocessed_image_width" and "preprocessed_image_height" in the '
                            'topology replacement configuration file in the "custom_attributes" section of the '
                            '"PreprocessorReplacement" replacer. This value is defined in the configuration file '
                            'samples/configs/*.config of the model in the Object Detection model zoo as '
                            '"min_dimension".')
            else:
                graph.graph['preprocessed_image_width'] = config_attrs['preprocessed_image_width']
                graph.graph['preprocessed_image_height'] = config_attrs['preprocessed_image_height']
        else:
            graph.graph['preprocessed_image_height'] = preprocessed_image_height_width[0]
            graph.graph['preprocessed_image_width'] = preprocessed_image_height_width[1]

        initial_input_node_name = 'image_tensor'
        if initial_input_node_name not in graph.nodes():
            raise Error('Input node "{}" of the graph is not found. Do not run the Model Optimizer with '
                        '"--input" command line parameter.'.format(initial_input_node_name))
        placeholder_node = Node(graph, initial_input_node_name)

        if placeholder_node.shape[0] != 1 and placeholder_node.shape[0] != -1:
            raise Error('The faster R-CNN model support batch size 1 only.')
        placeholder_node.shape[0] = 1  # batch size 1 is supported only
        placeholder_node.shape[1] = graph.graph['preprocessed_image_height']
        placeholder_node.shape[2] = graph.graph['preprocessed_image_width']

        to_float_node = placeholder_node.out_node(0)
        if not to_float_node.has('op') or to_float_node.op != 'Cast':
            raise Error('The output of the "{}" is not Cast operation. Cannot apply replacer.'.format(
                initial_input_node_name))

        # connect to_float_node directly with node performing scale on mean value subtraction
        if mul_node is None:
            create_edge(to_float_node, sub_node, 0, 0)
        else:
            create_edge(to_float_node, mul_node, 0, 1)

        print('The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if'
              ' applicable) are kept.')
        return {}

    @staticmethod
    def get_preprocessed_image_size_from_model(graph: nx.MultiDiGraph):
        """
        The function looks for nodes in the Preprocessor block with specific names for resized image shape. If one of
        the nodes exist return the desired size. If nodes do not exist then return None.
        :param graph: graph to operate on.
        :return: the tuple with height and width of the preprocessed image.
        """
        preprocess_resize_to_range_size_node_name = 'Preprocessor/map/while/ResizeToRange/Const'
        preprocess_resize_bilinear_node_name = 'Preprocessor/map/while/ResizeImage/ResizeBilinear'
        result = None
        if preprocess_resize_to_range_size_node_name in graph.nodes():
            preprocess_size_node = Node(graph, preprocess_resize_to_range_size_node_name)
            result = (int(preprocess_size_node.value.item()), int(preprocess_size_node.value.item()))
        elif preprocess_resize_bilinear_node_name in graph.nodes():
            preprocess_size_node = Node(graph, preprocess_resize_bilinear_node_name)
            result = (int(preprocess_size_node.in_node(1).value[0]), int(preprocess_size_node.in_node(1).value[1]))
        return result
