Legacy Model Optimizer Extensibility
====================================



.. toctree::
   :maxdepth: 1
   :hidden:

   legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification
   legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions
   legacy-model-optimizer-extensibility/[legacy]-extending-model-optimizer-with-caffe-python-layers

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated TensorFlow conversion method. The guide on the new and recommended method, using a new frontend, can be found in the  :doc:`Frontend Extensions <../../openvino-extensibility/frontend-extensions>` article.

This article describes Model Optimizer internals. Altering them may result in application instability, and in case of future changes to the API, lack of backward compatibility.

.. note::
   If you want to add support for ONNX, TensorFlow Lite, PaddlePaddle or TensorFlow operations, or you are not familiar with other extension alternatives in OpenVINO, read :doc:`this guide <../../openvino-extensibility>` instead.

.. _model-optimizer-extensibility:

Model Optimizer extensibility mechanism enables support of new operations and custom transformations to generate the optimized intermediate representation (IR) as described :doc:`here <../../openvino-ir-format/operation-sets>`.
This mechanism is a core part of Model Optimizer, as a huge set of examples showing how to add custom logic to support your model.

There are several cases when the customization is needed:

* A model contains operation(s) not known for the Model Optimizer, but these operation(s) could be expressed as a combination of supported operations. In this case, a custom transformation should be implemented to replace unsupported operation(s) with supported ones.
* A model contains a sub-graph of operations that can be replaced with a smaller number of operations to get better performance. This example corresponds to so-called *fusing transformations* (e.g., replacing a sub-graph performing the calculation :math:`x/(1.0+e^{-(beta*x)})` with a single operation of type :doc:`Swish <../../openvino-ir-format/operation-sets/operation-specs/activation/swish-4>`.
* A model contains a custom framework operation (the operation that is not a part of an official operation set of the framework) that was developed using the framework extensibility mechanism. In this case, Model Optimizer should know how to handle the operation and generate a corresponding section in an IR for it.

It is necessary to figure out how Model Optimizer represents a model in a memory and converts it to an IR before
going into details of the Model Optimizer extensibility mechanism.

.. note::
   All paths in this article are provided relatively to the Model Optimizer installation directory if not stated otherwise.

.. _mo_model_representation_in_memory:

==============================
Model Representation in Memory
==============================

The model can be represented as a directed graph, where nodes are operations and edges correspond to data passing from a
producer operation (node) to a consumer operation (node).

Model Optimizer uses Python class ``mo.graph.graph.Graph`` instance to represent the computation graph in memory during
the model conversion. This class is inherited from the ``networkx.MultiDiGraph`` class of the standard ``networkx`` Python
library. It provides many convenient methods to traverse and modify the graph. Refer to the ``mo/graph/graph.py`` file for examples.

Model Optimizer keeps all necessary information about the operation in node attributes. Model Optimizer uses the ``mo.graph.graph.Node`` class defined in the  ``mo/graph/graph.py`` file, which is a wrapper on top of a ``networkx`` node attributes
dictionary, and provides many convenient methods to work with the node. For example, the node ``my_node`` attribute with a
name ``my_attr`` can be retrieved from the node with the following code ``my_node.my_attr``, which is equivalent to obtaining
attribute with name ``my_attr`` in the ``graph.node[my_node]`` dictionary. For the class implementation details, refer to the ``mo/graph/graph.py`` file.

An operation may have several inputs and outputs. For example, operation :doc:`Split <../../openvino-ir-format/operation-sets/operation-specs/movement/split-1>` has
two inputs: data to split and axis to split along, and variable number of outputs depending on a value of attribute
``num_splits``. Each input data to the operation is passed to a specific operation **input port**. An operation produces
the output data from an **output port**. Input and output ports are numbered from 0 independently. Model Optimizer uses
classes ``mo.graph.port.Port`` and ``mo.graph.connection.Connection``, which are useful abstraction to perform graph
modifications like nodes connecting/re-connecting and graph traversing. These classes are widely used in the Model
Optimizer code so it is easy to find a lot of usage examples.

There is no dedicated class corresponding to an edge, so low-level graph manipulation is needed to get access to
edge attributes if needed. Meanwhile, most manipulations with nodes connections should be done with help of the
``mo.graph.connection.Connection`` and ``mo.graph.port.Port`` classes. Thus, low-level graph manipulation is error prone and
is strongly not recommended.

Further details and examples related to a model representation in memory are provided in the sections below, in a context
for a better explanation. For more information on how to use ports and connections, refer to the :doc:`Graph Traversal and Modification Using Ports and Connections <legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification>` article.

.. _mo_model_conversion_pipeline:

=========================
Model Conversion Pipeline
=========================

A model conversion pipeline can be represented with the following diagram:

.. image:: ../../../assets/images/MO_conversion_pipeline.svg

Each conversion step is reviewed in details below.

Model Loading
#############

Model Optimizer gets a trained model file as an input. The model loader component of Model Optimizer reads a model file
using Python bindings provided with the framework and builds an in-memory representation of a computation graph. There
is a separate loader for each supported framework. These loaders are implemented in the
``extensions/load/<FRAMEWORK>/loader.py`` files of Model Optimizer.

.. note::
   Model Optimizer uses a special parser for Caffe models built on top of the ``caffe.proto`` file. In the case of a model loading failure, Model Optimizer throws an error and requests preparation of the parser that can read the model. For more information on how to prepare the custom Caffe parser, refer to the :ref:`question #1 <question-1>` in the :doc:`Model Optimizer FAQ <legacy-conversion-api/[legacy]-model-optimizer-faq>`.

The result of a model loading step is a ``Graph`` object, which can be depicted like in the following example:

.. image:: ../../../assets/images/MO_graph_after_loader.svg

Model Optimizer loader saves an operation instance framework description (usually it is a Protobuf message) into a node
attribute usually with a name ``pb`` for each operation of an input model. It is important that this is a
**framework-specific** description of an operation. This means that an operation (e.g.
:doc:`Convolution <../../openvino-ir-format/operation-sets/operation-specs/convolution/convolution-1>` may be represented differently in, for example, Caffe and
TensorFlow frameworks but performs the same calculations from a mathematical point of view.

In the image above, the **Operation 2** has one input and two outputs. The tensor produced from the output **port 0** is
consumed with the **Operation 5** (the input **port 0**) and **Operation 3** (the input **port 1**). The tensor produced from the
output **port 1** is consumed with the **Operation 4** (the input **port 0**).

Each edge has two attributes: ``in`` and ``out``. They contain the input port number of the consumer node and the output port
number of the producer node. These attributes describe the fact that nodes are operations consuming some input tensors
and producing some output tensors. From the perspective of Model Optimizer, nodes themselves are **black boxes** because
they do not contain required information about the operation they perform.

Operations Attributes Extracting
################################

The next step is to parse framework-dependent operation representation saved in a node attribute and update the node
attributes with the operation specific attributes. There are three options to do this.

1.  The extractor extension approach (recommended way to extract attributes for an operation). Explained in details in the :doc:`Operation Extractor <legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions/[legacy]-optimizer-extractor>` article.
2.  The legacy approach with a built-in extractor. The ``mo/front/<FRAMEWORK>/extractor.py`` file (for example, the one for Caffe) defines a dictionary with extractors for specific operation types. A key in the dictionary is a type of an operation to trigger the extracting function for and the value is the function. The function has one parameter – a node to extract attributes from. This is a legacy and non-extensible approach so it should be avoided. This mechanism will be removed in future versions of Model Optimizer.

The extractors execution order is the following:

* ``CustomLayersMapping.xml`` (for Caffe models only).
* Model Optimizer extension.
* Built-in Model Optimizer extractor.

The result of operations attributes extracting step can be depicted like in the following example:

.. image:: ../../../assets/images/MO_graph_after_extractors.svg

The only difference in the graph from the previous step is that nodes contain dictionary with extracted attributes and
operation-specific attributes needed for Model Optimizer. However, from this step, Model Optimizer does not
need the original representation of the operation/model and just uses Model Optimizer representation (there are some
peculiar cases in which Model Optimizer still uses the ``pb`` attribute, covered in this
article partially). A detailed list of common node attributes and their values is provided in the
:doc:`Model Optimizer Operation <legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions/[legacy]-model-optimizer-operation>` article.

Front Phase
###########

For legacy reasons, you must specify shapes for all not fully-defined inputs of the model. In contrast, other
machine learning frameworks, like TensorFlow, let you create a model with undefined or partially defined input shapes.
As an example, undefined dimension is marked with an integer value ``-1`` in a TensorFlow model or has some string name
in an ONNX model.

During the front phase, Model Optimizer knows shape of the model inputs and constants only and does not know shapes
(and even ranks) of the intermediate tensors. But information about shapes may not be needed to implement particular
transformation. For example, the transformation ``extensions/front/TopKNormalize.py`` removes an attribute ``k``  from a
``TopK`` node and adds an input constant with the value ``k``. The transformation is needed to convert a ``TopK`` operation.
It comes from frameworks, where a number of output elements is defined as an attribute of the operation to the
OpenVINO :doc:`TopK <../../openvino-ir-format/operation-sets/operation-specs/sort/top-k-3>` operation semantic, which requires this value to be a separate input.

It is important to mention that sometimes it seems like transformation cannot be implemented during the front phase
because the actual values of inputs or shapes are needed. In fact, manipulations of shapes or values can be implemented
using operations that are added to the graph. Consider the
``extensions/front/onnx/flattenONNX_to_reshape.py`` transformation, which replaces an ONNX
`Flatten <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten>`__ operation with a sub-graph of operations performing
the following (when ``axis`` is not equal to 0 and 1):

1. Calculate a shape of the ``Flatten`` input tensor, using the :doc:`ShapeOf <../../openvino-ir-format/operation-sets/operation-specs/shape/shape-of-3>` operation.
2. Get the first ``axis`` elements from the output of ``Shape`` operation and calculate their product, using the :doc:`ReduceProd <../../openvino-ir-format/operation-sets/operation-specs/reduction/reduce-prod-1>` operation.
3. Concatenate output of the ``ReduceProd`` and constant with the value of ``-1`` (for an explanation of this value refer to the :doc:`Reshape <../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1>` specification page).
4. Use the concatenated value as the second input to the ``Reshape`` operation.

It is highly recommended to write shape-agnostic transformations to avoid model reshape-ability issues. For more information related to the reshaping of a model, refer to the :doc:`Using Shape Inference <../../../openvino-workflow/running-inference/changing-input-shape>` guide.

More information on how to develop front phase transformations and dedicated API description is provided in the
:ref:`Front Phase Transformations <mo_front_phase_transformations>`.

.. _mo_partial_inference:

Partial Inference
#################

Model Optimizer performs a partial inference of a model during model conversion. This procedure includes output shapes
calculation of all operations in a model and constant folding (value calculation for constant sub-graphs). The constant
folding is needed for the shape inference because in some cases evaluation of constant sub-graph is needed to calculate
output shapes. For example, the output shape for the :doc:`Reshape <../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1>` operation may be
defined as a mathematical expression using the :doc:`ShapeOf <../../openvino-ir-format/operation-sets/operation-specs/shape/shape-of-3>` operation output.

.. note::
   Model Optimizer does not fold sub-graphs starting from the :doc:`ShapeOf <../../openvino-ir-format/operation-sets/operation-specs/shape/shape-of-3>` operation by default because this leads to a model non-reshape-ability (the command-line parameter ``--static_shape`` can override this behavior). For more information related to reshaping of a model, refer to the :doc:`Using Shape Inference <../../../openvino-workflow/running-inference/changing-input-shape>` guide.

Model Optimizer calculates output shapes for all operations in a model to write them to Intermediate Representation files.

.. note::
   This is a legacy requirement. Starting with IR version 10, OpenVINO Runtime needs to know shapes of the :doc:`Const <../../openvino-ir-format/operation-sets/operation-specs/infrastructure/constant-1>` and the :doc:`Parameter <../../openvino-ir-format/operation-sets/operation-specs/infrastructure/parameter-1>` operations only. The OpenVINO Runtime calculates output shapes for all operations in a model, using shapes of :doc:`Parameter <../../openvino-ir-format/operation-sets/operation-specs/infrastructure/parameter-1>` and :doc:`Const <../../openvino-ir-format/operation-sets/operation-specs/infrastructure/constant-1>` operations defined with respective operation attributes.

Model Optimizer inserts **data** nodes to the computation graph before starting the partial inference phase. The data node
corresponds to the specific tensor produced with the operation. Each data node contains two attributes: ``shape``,
containing the shape of the tensor, and ``value``, which may contain the actual value of the tensor. The value for a ``value``
attribute is equal to ``None`` if this tensor value cannot be calculated. This happens in two cases: when a tensor value
depends on a values passed to the :doc:`Parameter <../../openvino-ir-format/operation-sets/operation-specs/infrastructure/parameter-1>` operation of a model or
Model Optimizer does not have value propagation implementation for the operation.

Before running partial inference, the graph can be depicted like in the following example:

.. image:: ../../../assets/images/MO_graph_before_partial_inference.svg

The difference in a graph structure with a graph during the front phase is not only in the data nodes, but also in the
edge attributes. Note that an ``out`` attribute is specified for edges **from operation** nodes only, while an ``in``
attribute is specified for edges **from data** nodes only. This corresponds to the fact that a tensor (data node) is
produced from a specific output port of an operation and is consumed with a specific input port of an operation. Also,
a unique data node is created for each output port of an operation. The node may be used as an input node for several
operation nodes. Similarly to the data node **data2_0**, which is consumed with the input **port 1** of the **Operation 3** and
input **port 0** of the **Operation 5**.

Now, consider how Model Optimizer performs shape and value propagation. Model Optimizer performs graph nodes
topological sort. An error message is thrown if a graph contains a cycle. Then, shape inference functions are called for
each node in the graph, according to the topological order. Each node of the graph must have an attribute called ``infer``
with a shape inference function, which is a function with one parameter – an instance of the ``Node`` class. The ``infer``
attribute is usually set in the operation extractor or when a node is added in some transformation using the Model
Optimizer operation class inherited from the ``mo.pos.Op`` class. For more information on how to specify a shape inference function,
refer to the :doc:`Model Optimizer Operation <legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions/[legacy]-model-optimizer-operation>` and :doc:`Operation Extractor <legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions/[legacy]-optimizer-extractor>` articles.

A shape inference function should calculate an operation (node) output shape(s) based on input shape(s) and operation
(node) attribute(s) and update ``shape`` and optionally ``value`` attributes of the corresponding data node(s). A simplified
example of the shape infer function for the :doc:`Reshape <../../openvino-ir-format/operation-sets/operation-specs/shape/reshape-1>` operation (the full version is
available in the ``mo/ops/reshape.py`` file):

.. code-block:: py
   :force:

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()  # get the input tensor shape
        new_shape = node.in_port(1).data.get_value()  # get the value defining the output tensor shape. This tensor may
                                                      # have special values like 0 and -1

        output_shape = ... # calculate output shape without special values like 0 and -1

        if node.in_port(0).data.get_value() is not None:  # if the input value is defined then calculate output value;
                                                          # shape will be updated automatically with the value shape
            node.out_port(0).data.set_value(node.in_port(0).data.get_value().reshape(output_shape))
        else:  # in the opposite case calculate the output shape only
            node.out_port(0).data.set_shape(output_shape)

Methods ``in_port()`` and ``output_port()`` of the ``Node`` class are used to get and set data node attributes. For more information on
how to use them, refer to the :doc:`Graph Traversal and Modification Using Ports and Connections <legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification>` article.

.. note::
   A shape inference function should perform output shape calculation in the original model layout. For example, OpenVINO™ supports Convolution operations in NCHW layout only but TensorFlow supports NHWC layout as well. Model Optimizer shape inference function calculates output shapes for NHWC Convolutions in NHWC layout and only during the layout change phase the shape is converted to NCHW.

.. note::
   There is a legacy approach to read data node attribute, like ``input_shape = op_node.in_node(0).shape`` and modify data nodes attributes, like ``op_node.out_node(0).shape = some_value``. This approach is still used in the Model Optimizer code but is not recommended. Instead, use the approach described in the :ref:`Ports <mo_intro_ports>`.

Middle Phase
############

The middle phase starts after partial inference. At this phase, a graph contains data nodes and output shapes of all
operations in the graph have been calculated. Any transformation implemented at this stage must update the ``shape``
attribute for all newly added operations. It is highly recommended to use API described in the
:doc:`Graph Traversal and Modification Using Ports and Connections <legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification>` because modification of a graph using this API causes automatic re-inference of affected nodes as well as necessary data nodes creation.

More information on how to develop middle transformations and dedicated API description is provided in the
:ref:`Middle Phase Transformations <mo_middle_phase_transformations>`.

NHWC to NCHW Layout Change
##########################

There are several middle transformations responsible for changing model layout from NHWC to NCHW. These transformations are triggered by default for TensorFlow models as TensorFlow supports Convolution operations in the NHWC layout.

This layout change is disabled automatically if the model does not have operations that OpenVINO™ needs to execute in the NCHW layout, for example, Convolutions in NHWC layout.

For more details on how it works, refer to the source code of the transformations mentioned in the below summary of the process:

1. Model Optimizer changes output shapes of most of operations producing 4D and 5D (four dimensional and five dimensional) tensors as if they were in NHWC layout to NCHW layout: ``nchw_shape = np.array(nhwc_shape)[0, 3, 1, 2]`` for 4D and ``nchw_shape = np.array(nhwc_shape)[0, 4, 1, 2, 3]`` for 5D. This permutation does not happen for some operations with specific conditions identified during a model conversion.
2. Model Optimizer inserts :doc:`Gather <../../openvino-ir-format/operation-sets/operation-specs/movement/gather-1>` operations to the sub-graph relates to shapes calculation in order to perform shape calculation in a correct layout.
3. Model Optimizer inserts :doc:`Transpose <../../openvino-ir-format/operation-sets/operation-specs/movement/transpose-1>` operations for some operations with specific conditions, identified during a model conversion, to produce correct inference results.

The main transformations responsible for a layout change are:

* ``extensions/middle/ApplyPermutations.py``
* ``extensions/middle/InsertLayoutPropagationTransposes.py``
* ``extensions/middle/MarkSubgraphsWithCorrectLayout.py``
* ``extensions/middle/ApplyNHWCtoNCHWpermutation.py``
* ``extensions/middle/LayoutChangeForConstantShapePaths.py``

Back Phase
##########

The back phase starts after the layout change to NCHW. This phase contains mostly the following transformations:

1. Transformations that should work with a graph in the NCHW layout and thus cannot be implemented in the middle phase.
2. Transformations that replace nodes corresponding to internal Model Optimizer operations with nodes corresponding to the :doc:`opset <../../openvino-ir-format/operation-sets/available-opsets>` operations.
3. Transformations that normalize operations inputs according to the specification.
4. Final optimization transformations.

A graph structure during the back phase is the same as during the middle phase. There is no difference in writing middle
and back transformations.

More information on how to develop back transformations and dedicated API description is provided in the
:ref:`Back Phase Transformations <mo_back_phase_transformations>`.

Intermediate Representation Emitting
####################################

The last phase of a model conversion is the Intermediate Representation emitting. Model Optimizer performs the following
steps:

1. Iterates over all operation nodes in the graph and checks that all nodes have the ``type`` attribute set. This attribute defines the operation type and is used in the OpenVINO to instantiate proper operation from the :doc:`opset <../../openvino-ir-format/operation-sets/available-opsets>` specified in the ``version`` attribute of the node. If a node does not have attribute ``type`` or its value is equal to ``None``, Model Optimizer exits with an error.
2. Performs type inference of graph operations similar to the shape inference. Inferred data types are saved to a port attributes in the IR.
3. Performs topological sort of the graph and changes ``id`` attribute of all operation nodes to be sequential integer values starting from 0.
4. Saves all Constants values to the ``.bin`` file. Constants with the same value are shared among different operations.
5. Generates an ``.xml`` file defining a graph structure. The information about operation inputs and outputs are prepared uniformly for all operations regardless of their type. A list of attributes to be saved to the ``.xml`` file is defined with the ``backend_attrs()`` or ``supported_attrs()`` of the ``Op`` class used for a graph node instantiation. For more information on how the operation attributes are saved to XML, refer to the function ``prepare_emit_ir()`` in the ``mo/pipeline/common.py`` file and :doc:`Model Optimizer Operation <legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions/[legacy]-model-optimizer-operation>` article.

====================
Additional Resources
====================

* :doc:`Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ <../../openvino-ir-format/operation-sets>`
* :doc:`Converting a Model to Intermediate Representation (IR) <legacy-conversion-api/[legacy]-setting-input-shapes>`
* :doc:`OpenVINO Model Representation <../../../openvino-workflow/running-inference/integrate-openvino-with-your-application/model-representation>`
* :doc:`OpenVINO™ Extensibility Mechanism <../../openvino-extensibility>`
* :doc:`Graph Traversal and Modification Using Ports and Connections <legacy-model-optimizer-extensibility/[legacy]-graph-traversal-and-modification>`
* :doc:`Model Optimizer Extensions <legacy-model-optimizer-extensibility/[legacy]-model-optimizer-extensions>`
* :doc:`Extending Model Optimizer with Caffe Python Layers <legacy-model-optimizer-extensibility/[legacy]-extending-model-optimizer-with-caffe-python-layers>`

