[LEGACY] Graph Transformation Extensions
==========================================

.. meta::
  :description: Learn about various base classes for front, middle and back phase
                transformations applied during model conversion with Model Optimizer.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated TensorFlow conversion method. The guide on the new and recommended method, using a new frontend, can be found in the  :doc:`Frontend Extensions <../../../../openvino-extensibility/frontend-extensions>` article.

Model Optimizer provides various base classes to implement :ref:`Front Phase Transformations <mo_front_phase_transformations>`,
:ref:`Middle Phase Transformations <mo_middle_phase_transformations>`, and :ref:`Back Phase Transformations <mo_back_phase_transformations>`.
All classes have the following common class attributes and methods:

1. The ``enabled`` attribute specifies whether the transformation is enabled or not. The value can be changed during runtime to enable or disable execution of the transformation during a model conversion. Default value is ``True``.
2. The ``id`` attribute specifies a unique transformation string identifier. This transformation identifier can be used to enable (disable) the transformation by setting environment variable ``MO_ENABLED_TRANSFORMS`` (``MO_DISABLED_TRANSFORMS``) with a comma separated list of ``ids``. The environment variables override the value of the ``enabled`` attribute of the transformation. Instead of using ``id`` attribute value you can add fully defined class name to ``MO_ENABLED_TRANSFORMS`` (``MO_DISABLED_TRANSFORMS``) variable, ``extensions.back.NonmalizeToNormalizeL2.NormalizeToNormalizeL2`` for example. It is an optional attribute.
3. The ``run_not_recursively`` attribute specifies whether the transformation should be executed in the sub-graphs, for example, body of the :doc:`TensorIterator <../../../../openvino-ir-format/operation-sets/operation-specs/infrastructure/tensor-iterator-1>` and the :doc:`Loop <../../../../openvino-ir-format/operation-sets/operation-specs/infrastructure/loop-5>`. Default value is ``True``.
4. The ``force_clean_up`` attribute specifies whether the graph clean up should be executed after the transformation. The graph cleanup removes nodes of the graph not reachable from the model inputs. Default value is ``False``.
5. The ``force_shape_inference`` attribute specifies whether the nodes marked with ``need_shape_inference`` attribute equal to ``True`` should be re-inferred after the transformation. Model Optimizer sets this attribute automatically for nodes, input(s) of which were changed during the transformation, or you can set this attribute manually in the transformation for the specific nodes. Default value is ``False``.
6. Attribute ``graph_condition`` specifies a list of functions with one parameter -- ``Graph`` object. The transformation is executed if and only if all functions return ``True``. If the attribute is not set, no check is performed.
7. Method ``run_before()`` returns a list of transformation classes which this transformation should be executed before.
8. Method ``run_after()`` returns a list of transformation classes which this transformation should be executed after.

.. note::
   Some of the transformation types have specific class attributes and methods, which are explained in the corresponding sections of this document.

Model Optimizer builds a graph of dependencies between registered transformations and executes them in the topological
order. To execute the transformation during a proper model conversion phase, Model Optimizer defines several
anchor transformations that do nothing. All transformations are ordered with respect to these anchor transformations.
The diagram below shows anchor transformations, some of built-in transformations and dependencies between them:

.. image:: ../../../../../assets/images/MO_transformations_graph.svg

User-defined transformations are executed after the corresponding ``Start`` and before the corresponding ``Finish`` anchor
transformations by default (if ``run_before()`` and ``run_after()`` methods have not been overridden).

.. note::
   The ``PreMiddleStart`` and ``PostMiddleStart`` anchors were introduced due to historical reasons to refactor the Model Optimizer pipeline, which initially had a hardcoded order of transformations.

.. _mo_front_phase_transformations:

===========================
Front Phase Transformations
===========================

There are several types of a front phase transformation:

1. :ref:`Pattern-Defined Front Phase Transformations <pattern_defined_front_phase_transformations>` triggered for each sub-graph of the original graph isomorphic to the specified pattern.
2. :ref:`Specific Operation Front Phase Transformations <specific_operation_front_phase_transformations>` triggered for the node with a specific ``op`` attribute value.
3. :ref:`Generic Front Phase Transformations <generic_front_phase_transformations>`.
4. Manually enabled transformation, defined with a JSON configuration file (for TensorFlow, ONNX, and PaddlePaddle models), specified using the ``--transformations_config`` command-line parameter:

   1. :ref:`Node Name Pattern Front Phase Transformations <node_name_pattern_front_phase_transformations>`.
   2. :ref:`Front Phase Transformations Using Start and End Points <start_end_points_front_phase_transformations>`.
   3. :ref:`Generic Front Phase Transformations Enabled with Transformations Configuration File <generic_transformations_config_front_phase_transformations>`.

.. _pattern_defined_front_phase_transformations:

Pattern-Defined Front Phase Transformations
###########################################

This type of transformation is implemented using ``mo.front.common.replacement.FrontReplacementSubgraph`` and
``mo.front.common.replacement.FrontReplacementPattern`` as base classes and works as follows:

1. Define a sub-graph to be matched, using a list of nodes with attributes and edges connecting them (edges may also have attributes).
2. Model Optimizer searches for all sub-graphs of the original graph, isomorphic to the specified sub-graph (pattern).
3. Model Optimizer executes the defined function performing graph transformation for each instance of a matched sub-graph. You can override different functions in the base transformation class so the Model Optimizer works differently:

   1. The ``replace_sub_graph(self, graph, match)`` override the method. In this case Model Optimizer only executes the overridden function, pass the ``graph`` object and a dictionary describing the matched sub-graph. You are required to write the transformation and connect the newly created nodes to the rest of the graph.
   2. The ``generate_sub_graph(self, graph, match)`` override the method. This case is not recommended for use because it is the most complicated approach. It can be effectively replaced with one of two previous approaches.

The sub-graph pattern is defined in the ``pattern()`` function. This function should return a dictionary with two keys:
``nodes`` and ``edges``:

* The value for the ``nodes`` key is a list of tuples with two elements.

  * The first element is an alias name for a node that will be used to define edges between nodes and in the transformation function.
  * The second element is a dictionary with attributes. The key is a name of an attribute that should exist in the node. The value for the attribute can be some specific value to match or a function that gets a single parameter - the attribute value from the node. The function should return the result of attribute comparison with a dedicated value.

* The value for the ``edges`` key is a list of tuples with two or three elements.

  * The first element is the alias name of the node producing a tensor.
  * The second element is the alias name of the node consuming the tensor.
  * The third element (optional) is the dictionary with expected edge attributes. This dictionary usually contains attributes like ``in`` and ``out``, defining input and output ports.

Consider the example of a front transformation implemented in the ``extensions/front/Mish_fusion.py`` file performing
fusing of the sub-graph defining the :doc:`Mish <../../../../openvino-ir-format/operation-sets/operation-specs/activation/mish-4>` activation function into a single
operation:

.. code-block:: py
   :force:

   from openvino.tools.mo.front.Softplus_fusion import SoftplusFusion
   from openvino.tools.mo.ops.activation_ops import Mish
   from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
   from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
   from openvino.tools.mo.graph.graph import Graph, rename_nodes


   class MishFusion(FrontReplacementSubgraph):
       """
       The transformation looks for the pattern with Softplus defining the Mish function: Mish(x) = x * tanh(SoftPlus(x)).
       """
       enabled = True  # Transformation is enabled.

       def run_after(self):  # Run this transformation after "SoftplusFusion" transformation.
           return [SoftplusFusion]

       def pattern(self):  # Define pattern according to formulae x * tanh(SoftPlus(x)).
           return dict(
               nodes=[
                   ('mul', dict(op='Mul')),
                   ('tanh', dict(op='Tanh')),
                   ('softplus', dict(op='SoftPlus')),
               ],
               edges=[
                   ('softplus', 'tanh'),
                   ('tanh', 'mul'),
               ])

       def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):  # Entry point for the transformation.
           mul = match['mul']  # Get the Node corresponding to matched "mul" node.
           mul_name = mul.soft_get('name', mul.id)
           softplus = match['softplus']  # Get the Node corresponding to the matched "softplus" node.

           # Determine the input port of Mul which gets the 'input' node output.
           input_port_idx = int(mul.in_port(0).get_connection().get_source().node.soft_get('op') == 'Tanh')

           # Check that the same tensor is provided as input to Mul and SoftPlus.
           if mul.in_port(input_port_idx).get_source() != softplus.in_port(0).get_source():
               return

           mish = Mish(graph, {}).create_node()  # Create Mish operation.
           mish.in_port(0).connect(mul.in_port(input_port_idx).get_source())  # Connect input to the Mish.
           mul.out_port(0).get_connection().set_source(mish.out_port(0))  # Reconnect outgoing edge from "mul" to Mish.

           # Rename the created Mish operation to have the name of the "mul" node, which produced the value equal to the
           # Mish output.
           rename_nodes([(mul, mul_name + '/TBR'), (mish, mul_name)])

.. _specific_operation_front_phase_transformations:

Specific Operation Front Phase Transformations
##############################################

This type of transformation is implemented using ``mo.front.common.replacement.FrontReplacementOp`` as base class and
works as follows:

1. Define an operation type to trigger the transformation.
2. Model Optimizer searches for all nodes in the graph with the attribute ``op`` equal to the specified value.
3. Model Optimizer executes the defined function performing graph transformation for each instance of a matched node. You can override different functions in the base transformation class and Model Optimizer works differently:

   1. The ``replace_sub_graph(self, graph, match)`` override method. In this case, Model Optimizer only executes the overridden function. Pass the ``graph`` object and a dictionary with a single key ``op`` with the matched node as value. You are required to write the transformation and connect the newly created nodes to the rest of the graph.
   2. The ``replace_op(self, graph, node)`` override method. In this case, Model Optimizer executes the overridden function. Pass the ``graph`` object and the matched node as ``node`` parameter. If the function returns an ``id`` of some node, then the ``Node`` with this ``id`` is connected to the consumers of the matched node. After applying the transformation, the matched node is removed from the graph.

The ``FrontReplacementOp`` class provides a simpler mechanism to match a single operation with specific value of the ``op``
(write the ``op`` attribute in the class instead of defining a ``pattern()`` function) attribute and perform the
transformation.

Consider an example transformation from the ``extensions/front/Pack.py`` file, which replaces ``Pack`` operation from
the TensorFlow:

.. code-block:: py
   :force:

   from openvino.tools.mo.front.common.partial_infer.utils import int64_array
   from openvino.tools.mo.front.common.replacement import FrontReplacementOp
   from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
   from openvino.tools.mo.graph.graph import Node, Graph, rename_nodes
   from openvino.tools.mo.ops.concat import Concat
   from openvino.tools.mo.ops.unsqueeze import Unsqueeze


   class Pack(FrontReplacementOp):
       op = "Pack"  # Trigger transformation for all nodes in the graph with the op = "Pack" attribute
       enabled = True  # Transformation is enabled.

       def replace_op(self, graph: Graph, node: Node):  # Entry point for the transformation.
           # Create a Concat operation with a number of inputs equal to a number of inputs to Pack.
           out_node = Concat(graph, {'axis': node.axis, 'in_ports_count': len(node.in_ports())}).create_node()
           pack_name = node.soft_get('name', node.id)

           for ind in node.in_ports():
               # Add dimension of size 1 to all inputs of the Pack operation and add them as Concat inputs.
               unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array([node.axis])},
                                                            {'name': node.soft_get('name', node.id) + '/Unsqueeze'})
               node.in_port(ind).get_connection().set_destination(unsqueeze_node.in_port(0))
               unsqueeze_node.out_port(0).connect(out_node.in_port(ind))

           # Rename the created Concat operation to have the name of the "pack" node, which produced the value equal to the
           # Concat output.
           rename_nodes([(node, pack_name + '/TBR'), (out_node, pack_name)])
           return [out_node.id]  # Reconnect the Pack operation consumers to get input from Concat instead.


.. _generic_front_phase_transformations:

Generic Front Phase Transformations
###################################

Model Optimizer provides a mechanism to implement generic front phase transformation. This type of transformation is
implemented using ``mo.front.common.replacement.FrontReplacementSubgraph`` or
``mo.front.common.replacement.FrontReplacementPattern`` as base classes. Make sure the transformation is enabled before trying to execute it.
Then, Model Optimizer executes the ``find_and_replace_pattern(self, graph)`` method and
provides a ``Graph`` object as an input.

Consider the example of a generic front transformation from the ``extensions/front/SqueezeNormalize.py`` file performing
normalization of the :doc:`Squeeze <../../../../openvino-ir-format/operation-sets/operation-specs/shape/squeeze-1>` operation. Older version of the operation had a list of
axes to squeeze as an attribute, but now it is a separate input. For backward compatibility, the Model Optimizer
operation supports both semantics. Before IR generation, however, the operation should be normalized according to the
specification.

.. code-block:: py
   :force:

   import logging as log

   from openvino.tools.mo.front.common.partial_infer.utils import int64_array
   from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
   from openvino.tools.mo.graph.graph import Graph
   from openvino.tools.mo.ops.const import Const
   from openvino.tools.mo.utils.error import Error


   class SqueezeNormalize(FrontReplacementPattern):
       """
       Normalizes inputs of the Squeeze layers. The layers should have two inputs: the input with data and input with the
       dimensions to squeeze. If the second input is omitted then all dimensions of size 1 should be removed.
       """
       enabled = True  # The transformation is enabled.

       def find_and_replace_pattern(self, graph: Graph):  # The function is called unconditionally.
           for squeeze_node in graph.get_op_nodes(op='Squeeze'):  # Iterate over all nodes with op='Squeeze'.
               # If the operation has only 1 input node and no 'squeeze_dims' Node attribute, then convert the attribute to
               # the operation input.
               if len(squeeze_node.in_nodes()) == 1 and squeeze_node.has_valid('squeeze_dims'):
                   dims_node = Const(graph, {'name': squeeze_node.id + '/Dims',
                                             'value': int64_array(squeeze_node.squeeze_dims)}).create_node()
                   squeeze_node.in_port(1).connect(dims_node.out_port(0))
                   del squeeze_node['squeeze_dims']
               # If two inputs already exist, that means the operation is already normalized.
               elif len(squeeze_node.in_nodes()) == 2:
                   log.debug('The Squeeze node "{}" is already normalized'.format(squeeze_node.name))
               # In all other cases, raise an error.
               else:
                   raise Error('The Squeeze layer "{}" should either have 2 inputs or one input and an "squeeze_dims" '
                               'attribute'.format(squeeze_node.soft_get('name')))

For the details on implementation and how these front phase transformations work, refer to the ``mo/front/common/replacement.py``
file.

.. _node_name_pattern_front_phase_transformations:

Node Name Pattern Front Phase Transformations
#############################################

TensorFlow uses a mechanism of scope to group related operation nodes. It is a good practice to put nodes performing
particular task into the same scope. This approach divides a graph into logical blocks that are easier to review in the
TensorBoard. The scope, in fact, just defines a common name prefix for the nodes belonging to it.

For example, Inception topologies contain several types of so-called **Inception blocks**. Some of them are equal to each
other, but located in different places of the network. For example, Inception V4 from the
`TensorFlow-Slim image classification model library <https://github.com/tensorflow/models/tree/master/research/slim>`__ has
``Mixed_5b``, ``Mixed_5c`` and ``Mixed_5d`` inception blocks with exactly the same nodes, with the same set of attributes.

Consider a situation when these Inception blocks are implemented extremely efficiently using a single Inference
Engine operation called ``InceptionBlock`` and these blocks in the model need to be replaced with instances of this operation.
Model Optimizer provides mechanism to trigger the transformation for a sub-graph of operations defined by the node name
regular expressions (scope). In this particular case, some of the patterns are: ``.*InceptionV4/Mixed_5b``,
``.*InceptionV4/Mixed_5c`` and ``.*InceptionV4/Mixed_5d``. Each pattern starts with ``.*``, because the ``InceptionV4`` prefix
is added to all nodes names during a model freeze.

This type of transformation is implemented using ``mo.front.tf.replacement.FrontReplacementFromConfigFileSubGraph`` as a
base class and works as follows:

1. Prepare a JSON configuration file template defining node names patterns.
2. Run Model Optimizer with the ``--tensorflow_custom_operations_config_update`` command-line parameter, and Model Optimizer adds information about input and output nodes of the specified sub-graphs.
3. Model Optimizer executes the defined transformation **only** when you specify the path to the configuration file updated in step 2 using the ``--transformations_config`` command-line parameter.

Consider the following possible configuration file template for the Inception Block transformation:

.. code-block:: json

   [
       {
           "custom_attributes": {
               "attr1_key": "attr1_value",
               "attr2_key": 123456
           },
           "id": "InceptionBlockTransformation",
           "instances": [
               ".*InceptionV4/Mixed_5b",
               ".*InceptionV4/Mixed_5c",
               ".*InceptionV4/Mixed_5d"
           ],
           "match_kind": "scope"
       }
   ]

The configuration file contains a list of dictionaries. Each dictionary defines one transformation. Each transformation
is defined with several parameters:

* ``id`` - **(Mandatory)** — is a unique identifier of the transformation. It is used in the Python code that implements the transformation to link the class and the transformation description from the configuration file.
* ``match_kind`` - **(Mandatory)** —  is a string that specifies the matching algorithm. For the node name pattern case, the value should be equal to ``scope``. Another possible values are described in the dedicated sections below.
* ``instances`` - **(Mandatory)** —  specifies instances of the sub-graph to be matched. It contains a list of node names prefixes patterns for the match kind of the ``scope`` type.
* ``custom_attributes`` - **(Optional)** —  is a dictionary with attributes that can be used in the transformation code.

After running Model Optimizer with additional ``--tensorflow_custom_operations_config_update`` parameter pointing to
the template configuration file, the content of the file should be updated with two new sections ``inputs`` and ``outputs``.
The file content after the update is as follows:

.. code-block:: json

   [
       {
           "id": "InceptionBlockTransformation",
           "custom_attributes": {
               "attr1_key": "attr1_value",
               "attr2_key": 123456
           },
           "instances": [
               ".*InceptionV4/Mixed_5b",
               ".*InceptionV4/Mixed_5c",
               ".*InceptionV4/Mixed_5d"
           ],
           "match_kind": "scope",
           "inputs": [
               [
                   {
                       "node": "Branch_2/Conv2d_0a_1x1/Conv2D$",
                       "port": 0
                   },
                   {
                       "node": "Branch_3/AvgPool_0a_3x3/AvgPool$",
                       "port": 0
                   },
                   {
                       "node": "Branch_1/Conv2d_0a_1x1/Conv2D$",
                       "port": 0
                   },
                   {
                       "node": "Branch_0/Conv2d_0a_1x1/Conv2D$",
                       "port": 0
                   }
               ]
           ],
           "outputs": [
               {
                   "node": "concat$",
                   "port": 0
               }
           ]
       }
   ]

The value for ``inputs`` key is a list of lists describing input tensors of the sub-graph. Each element of the top-level
list corresponds to one unique input tensor of the sub-graph. Each internal list describes a list of nodes consuming
this tensor and port numbers, where the tensor is consumed. Model Optimizer generates regular expressions for the input
nodes names to uniquely identify them in each instance of the sub-graph, defined by the ``instances``. Denote these nodes
as input nodes of the sub-graph.

In the InceptionV4 topology, the ``InceptionV4/Mixed_5b`` block has four input tensors from outside of the sub-graph,
but all of them are produced by the ``InceptionV4/Mixed_5a/concat`` node. Therefore, the top-level list of the ``inputs``
contains one list corresponding to this tensor. Four input nodes of the sub-graph consume the tensor produced by
``InceptionV4/Mixed_5a/concat`` node. In this case, all four input nodes consume input tensor into "port 0".

The order of items in the internal list describing nodes does not matter, but the order of elements in the top-level
list is important. This order defines how Model Optimizer attaches input tensors to a new generated
node if the sub-graph is replaced with a single node. The ``i``-th input node of the sub-graph is obtained using
``match.single_input_node(i)`` call in the sub-graph transformation code. More information about API is given below. If it is
necessary to change the order of input tensors, the configuration file can be edited in the text editor.

The value for the ``outputs`` key is a list describing nodes of the sub-graph producing tensor, that goes outside of the
sub-graph or does not have child nodes. Denote these nodes as output nodes of the sub-graph. The order of elements in
the list is important. The ``i``-th element of the list describes the ``i``-th output tensor of the sub-graph, which could be
obtained using ``match.output_node(i)`` call. The order of elements can be manually changed in the configuration file.
Model Optimizer uses this order to connect output edges if the sub-graph is replaced with a single node.

For more examples of this type of transformation, refer to the :doc:`Converting TensorFlow Object Detection API Models <../../legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-conversion-tutorials/convert-tensorflow-object-detection>` guide.

.. _start_end_points_front_phase_transformations:

Front Phase Transformations Using Start and End Points
######################################################

This type of transformation is implemented using ``mo.front.tf.replacement.FrontReplacementFromConfigFileSubGraph`` as a
base class and works as follows:

1. Prepare a JSON configuration file that defines the sub-graph to match, using two lists of node names: "start" and "end" nodes.
2. Model Optimizer executes the defined transformation **only** when you specify the path to the configuration file using the ``--transformations_config`` command-line parameter . Model Optimizer performs the following steps to match the sub-graph:

   1. Starts a graph traversal from every start node following the direction of the graph edges. The search stops in an end node or in the case of a node without consumers. All visited nodes are added to the matched sub-graph.
   2. Starts another graph traversal from each non-start node of the sub-graph, i.e. every node except nodes from the "start" list. In this step, the edges are traversed in the opposite edge direction. All newly visited nodes are added to the matched sub-graph. This step is needed to add nodes required for calculation values of internal nodes of the matched sub-graph.
   3. Checks that all "end" nodes were reached from "start" nodes. If not, it exits with an error.
   4. Checks that there are no :doc:`Parameter <../../../../openvino-ir-format/operation-sets/operation-specs/infrastructure/parameter-1>` operations among added nodes. If they exist, the sub-graph depends on the inputs of the model. Such configuration is considered incorrect so  Model Optimizer exits with an error.

This algorithm finds all nodes "between" start and end nodes and nodes needed for calculation of non-input nodes of the
matched sub-graph.

The example of a JSON configuration file for a transformation with start and end points is
``extensions/front/tf/ssd_support_api_v1.15.json``:

.. code-block:: json

   [
       {
           "custom_attributes": {
               "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",
               "pad_mode": "caffe.ResizeParameter.CONSTANT",
               "resize_mode": "caffe.ResizeParameter.WARP",
               "clip_before_nms": false,
               "clip_after_nms": true
           },
           "id": "ObjectDetectionAPISSDPostprocessorReplacement",
           "include_inputs_to_sub_graph": true,
           "include_outputs_to_sub_graph": true,
           "instances": {
               "end_points": [
                   "detection_boxes",
                   "detection_scores",
                   "num_detections"
               ],
               "start_points": [
                   "Postprocessor/Shape",
                   "Postprocessor/scale_logits",
                   "Postprocessor/Tile",
                   "Postprocessor/Reshape_1",
                   "Postprocessor/Cast_1"
               ]
           },
           "match_kind": "points"
       }
   ]

The format of the file is similar to the one provided as an example in the
:ref:`Node Name Pattern Front Phase Transformations <node_name_pattern_front_phase_transformations>` section. The difference is in
the value of the ``match_kind`` parameter, which should be equal to the ``points`` and the format of the ``instances`` parameter,
which should be a dictionary with two keys ``start_points`` and ``end_points``, defining start and end node names
respectively.

.. note::
   The ``include_inputs_to_sub_graph`` and ``include_outputs_to_sub_graph`` parameters are redundant and should be always equal to ``true``.

.. note::
   This sub-graph match algorithm has a limitation that each start node must have only one input. Therefore, it is not possible to specify, for example, the :doc:`Convolution <../../../../openvino-ir-format/operation-sets/operation-specs/convolution/convolution-1>` node as input because it has two inputs: data tensor and tensor with weights.

For other examples of transformations with points, refer to the
:doc:`Converting TensorFlow Object Detection API Models <../../legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-conversion-tutorials/convert-tensorflow-object-detection>` guide.

.. _generic_transformations_config_front_phase_transformations:

Generic Front Phase Transformations Enabled with Transformations Configuration File
###################################################################################

This type of transformation works similarly to the :ref:`Generic Front Phase Transformations <generic_front_phase_transformations>`
but require a JSON configuration file to enable it similarly to
:ref:`Node Name Pattern Front Phase Transformations <node_name_pattern_front_phase_transformations>` and
:ref:`Front Phase Transformations Using Start and End Points <start_end_points_front_phase_transformations>`.

The base class for this type of transformation is
``mo.front.common.replacement.FrontReplacementFromConfigFileGeneral``. Model Optimizer executes the
``transform_graph(self, graph, replacement_descriptions)`` method and provides the ``Graph`` object and dictionary with values
parsed from the `custom_attributes` attribute of the provided JSON configuration file.

The example of the configuration file for this type of transformation is ``extensions/front/tf/yolo_v1_tiny.json``:

.. code-block:: json

   [
     {
       "id": "TFYOLO",
       "match_kind": "general",
       "custom_attributes": {
         "classes": 20,
         "coords": 4,
         "num": 2,
         "do_softmax": 0
       }
     }
   ]

and the corresponding transformation file is ``./extensions/front/YOLO.py``:

.. code-block:: py
   :force:

   from openvino.tools.mo.front.no_op_eraser import NoOpEraser
   from openvino.tools.mo.front.standalone_const_eraser import StandaloneConstEraser
   from openvino.tools.mo.ops.regionyolo import RegionYoloOp
   from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
   from openvino.tools.mo.graph.graph import Node, Graph
   from openvino.tools.mo.ops.result import Result
   from openvino.tools.mo.utils.error import Error


   class YoloRegionAddon(FrontReplacementFromConfigFileGeneral):
       """
       Replaces all Result nodes in graph with YoloRegion->Result nodes chain.
       YoloRegion node attributes are taken from configuration file
       """
       replacement_id = 'TFYOLO'  # The identifier matching the "id" attribute in the JSON file.

       def run_after(self):
           return [NoOpEraser, StandaloneConstEraser]

       def transform_graph(self, graph: Graph, replacement_descriptions):
           op_outputs = [n for n, d in graph.nodes(data=True) if 'op' in d and d['op'] == 'Result']
           for op_output in op_outputs:
               last_node = Node(graph, op_output).in_node(0)
               op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1)
               op_params.update(replacement_descriptions)
               region_layer = RegionYoloOp(graph, op_params)
               region_layer_node = region_layer.create_node([last_node])
               # In here, 'axis' from 'dim_attrs' can be removed to avoid permutation from axis = 1 to axis = 2.
               region_layer_node.dim_attrs.remove('axis')
               Result(graph).create_node([region_layer_node])
               graph.remove_node(op_output)

The configuration file has only 3 parameters: ``id`` identifier of the transformation , ``match_kind`` (which should be equal
to ``general``) and the ``custom_attributes`` dictionary with custom attributes accessible in the transformation.

.. _mo_middle_phase_transformations:

============================
Middle Phase Transformations
============================

There are two types of middle phase transformations:

1. :ref:`Pattern-Defined Middle Phase Transformations <pattern_defined_middle_phase_transformations>` triggered for each sub-graph of the original graph, isomorphic to the specified pattern.
2. :ref:`Generic Middle Phase Transformations <generic_middle_phase_transformations>`.

.. _pattern_defined_middle_phase_transformations:

Pattern-Defined Middle Phase Transformations
############################################

This type of transformation is implemented using ``mo.middle.replacement.MiddleReplacementPattern`` as a base class and
works similarly to the :ref:`Pattern-Defined Middle Phase Transformations <pattern_defined_middle_phase_transformations>`
The are two differences:

1. The transformation entry function name is ``replace_pattern(self, graph, match)``.
2. The pattern defining the graph should contain data nodes because the structure of the graph is different between front and middle phases. For more information about the graph structure changes, refer to the :ref:`Partial Inference <mo_partial_inference>`.

For the example of a pattern-defined middle transformation, refer to the ``extensions/middle/L2NormToNorm.py`` file.

.. _generic_middle_phase_transformations:

Generic Middle Phase Transformations
####################################

Model Optimizer provides a mechanism to implement generic middle phase transformations. This type of transformation is
implemented using ``mo.middle.replacement.MiddleReplacementPattern`` as a base class and works similarly to the
:ref:`Generic Front Phase Transformations <generic_front_phase_transformations>`. The only difference is that the
transformation entry function name is ``find_and_replace_pattern(self, graph: Graph)``.

For the example of this transformation, refer to the ``extensions/middle/CheckForCycle.py`` file.

.. _mo_back_phase_transformations:

==========================
Back Phase Transformations
==========================

There are two types of back phase transformations:

1. :ref:`Pattern-Defined Back Phase Transformations <pattern_defined_back_phase_transformations>` triggered for each sub-graph of the original graph, isomorphic to the specified pattern.
2. :ref:`Generic Back Phase Transformations <generic_back_phase_transformations>`.

.. note::
   The graph layout during the back phase is always NCHW. However, during the front and middle phases it could be NHWC if the original model was using it. For more details, refer to :ref:`Model Conversion Pipeline <mo_model_conversion_pipeline>`.

.. _pattern_defined_back_phase_transformations:

Pattern-Defined Back Phase Transformations
##########################################

This type of transformation is implemented using ``mo.back.replacement.MiddleReplacementPattern`` as a base class and
works the same way as :ref:`Pattern-Defined Middle Phase Transformations <pattern_defined_middle_phase_transformations>`.

For the example of a pattern-defined back transformation, refer to the ``extensions/back/ShufflenetReLUReorder.py`` file.

.. _generic_back_phase_transformations:

Generic Back Phase Transformations
##################################

Model Optimizer provides mechanism to implement generic back phase transformations. This type of transformation is
implemented using ``mo.back.replacement.BackReplacementPattern`` as a base class and works the same way as
:ref:`Generic Middle Phase Transformations <generic_middle_phase_transformations>`.

For the example of this transformation, refer to the ``extensions/back/GatherNormalizer.py`` file.

====================
Additional Resources
====================

* :doc:`Model Optimizer Extensibility <../../legacy-model-optimizer-extensibility>`
* :doc:`Graph Traversal and Modification Using Ports and Connections <../../legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification>`
* :doc:`Model Optimizer Extensions <../[legacy]-model-optimizer-extensions>`
* :doc:`Extending Model Optimizer with Caffe Python Layers <../[legacy]-extending-model-optimizer-with-caffe-python-layers>`

