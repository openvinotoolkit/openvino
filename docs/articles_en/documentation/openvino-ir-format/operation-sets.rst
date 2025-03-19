Operation Sets in OpenVINO
==========================


.. meta::
  :description: Learn the essentials of representing deep learning models in OpenVINO
                IR format and the use of supported operation sets.



This article provides essential information on the format used for representation of deep learning models in OpenVINO toolkit and supported operation sets.

Overview of Artificial Neural Networks Representation
#####################################################

A deep learning network is usually represented as a directed graph describing the flow of data from the network input data to the inference results.
Input data can be in the form of images, video, text, audio, or preprocessed information representing objects from the target area of interest.

Here is an illustration of a small graph representing a model that consists of a single Convolutional layer and activation function:

.. image:: ../../assets/images/small_IR_graph_demonstration.png

Vertices in the graph represent layers or operation instances such as convolution, pooling, and element-wise operations with tensors.
The terms of "layer" and "operation" are used interchangeably within OpenVINO documentation and define how the input data is processed to produce output data for a node in a graph.
An operation node in a graph may consume data at one or multiple input ports.
For example, an element-wise addition operation has two input ports that accept tensors to be summed.
Some operations do not have any input ports, for example the ``Const`` operation which produces without any input.
An edge between operations represents data flow or data dependency implied from one operation node to another.

Each operation produces data on one or multiple output ports. For example, convolution produces an output tensor with activations at a single output port. The ``Split`` operation usually has multiple output ports, each producing part of an input tensor.

Depending on a deep learning framework, the graph can also contain extra nodes that explicitly represent tensors between operations.
In such representations, operation nodes are not connected to each other directly. They are rather using data nodes as intermediate stops for data flow.
If data nodes are not used, the produced data is associated with an output port of the corresponding operation node that produces the data.

A set of various operations used in a network is usually fixed for each deep learning framework.
It determines expressiveness and level of representation available in that framework.
Sometimes, a network that can be represented in one framework is hard or impossible to be represented in another one or should use significantly different graph, because operation sets used in those two frameworks do not match.

Operation Sets
##############

Operations in OpenVINO Operation Sets are selected based on capabilities of supported deep learning frameworks and hardware capabilities of the target inference device.
A set consists of several groups of operations:

* Conventional deep learning layers such as ``Convolution``, ``MaxPool``, and ``MatMul`` (also known as ``FullyConnected``).

* Various activation functions such as ``ReLU``, ``Tanh``, and ``PReLU``.

* Generic element-wise arithmetic tensor operations such as ``Add``, ``Subtract``, and ``Multiply``.

* Comparison operations that compare two numeric tensors and produce boolean tensors, for example, ``Less``, ``Equal``, ``Greater``.

* Logical operations that are dealing with boolean tensors, for example, ``And``, ``Xor``, ``Not``.

* Data movement operations which are dealing with parts of tensors, for example, ``Concat``, ``Split``, ``StridedSlice``, ``Select``.

* Specialized operations that implement complex algorithms dedicated for models of specific type, for example, ``DetectionOutput``, ``RegionYolo``, ``PriorBox``.

For more information, refer to the complete description of the supported operation sets in the :doc:`Available Operation Sets <operation-sets/available-opsets>` article.

How to Read Opset Specification
###############################

In the :doc:`Available Operation Sets <operation-sets/available-opsets>` there are opsets and there are operations.
Each opset specification has a list of links to operations descriptions that are included into that specific opset.
Two or more opsets may refer to the same operation.
That means an operation is kept unchanged from one operation set to another.

The description of each operation has a ``Versioned name`` field.
For example, the `ReLU` entry point in :doc:`opset1 <operation-sets/available-opsets/opset1>` refers to :doc:`ReLU-1 <operation-sets/operation-specs/activation/relu-1>` as the versioned name.
Meanwhile, `ReLU` in `opset2` refers to the same `ReLU-1` and both `ReLU` operations are the same operation and it has a single :doc:`description <operation-sets/operation-specs/activation/relu-1>`, which means that ``opset1`` and ``opset2`` share the same operation ``ReLU``.

To differentiate versions of the same operation type such as ``ReLU``, the ``-N`` suffix is used in a versioned name of the operation.
The ``N`` suffix usually refers to the first occurrence of ``opsetN`` where this version of the operation is introduced.
There is no guarantee that new operations will be named according to that rule. The naming convention might be changed, but not for old operations which are frozen completely.

IR Versions vs Operation Set Versions
######################################

The expressiveness of operations in OpenVINO is highly dependent on the supported frameworks and target hardware capabilities.
As the frameworks and hardware capabilities grow over time, the operation set is constantly evolving to support new models.
To maintain backward compatibility and growing demands, both IR format and operation set have versioning.

Version of IR specifies the rules which are used to read the XML and binary files that represent a model. It defines an XML schema and compatible operation set that can be used to describe operations.

Historically, there are two major IR version epochs:

1. The older one includes IR versions from version 1 to version 7 without versioning of the operation set. During that epoch, the operation set has been growing evolutionally accumulating more layer types and extending existing layer semantics. Changing of the operation set for those versions meant increasing of the IR version.

2. OpenVINO 2020.1 is the starting point of the next epoch. With IR version 10 introduced in OpenVINO 2020.1, the versioning of the operation set is tracked separately from the IR versioning. Also, the operation set was significantly reworked as the result of nGraph integration to the OpenVINO.

The first supported operation set in the new epoch is ``opset1``.
The number after ``opset`` is going to be increased each time new operations are added or old operations deleted at the release cadence.

The operations from the new epoch cover more TensorFlow and ONNX operations that better match the original operation semantics from the frameworks, compared to the operation set used in the older IR versions (7 and lower).

The name of the opset is specified for each operation in IR.
The IR version is specified once.
Here is an example from the IR snippet:

.. code-block:: cpp

   <?xml version="1.0" ?>
   <net name="model_file_name" version="10">  <!-- Version of the whole IR file is here; it is 10 -->
       <layers>
           <!-- Version of operation set that the layer belongs to is described in <layer>
               tag attributes. For this operation, it is version="opset1". -->
           <layer id="0" name="input" type="Parameter" version="opset1">
               <data element_type="f32" shape="1,3,32,100"/> <!-- attributes of operation -->
               <output>
                   <!-- description of output ports with type of element and tensor dimensions -->
                   <port id="0" precision="FP32">
                       <dim>1</dim>
                       <dim>3</dim>

                        ...

The ``type="Parameter"`` and ``version="opset1"`` attributes in the example above mean "use that version of the ``Parameter`` operation that is included in the ``opset1`` operation set. "

When a new operation set is introduced, most of the operations remain unchanged and are just aliased from the previous operation set within a new one.
The goal of operation set version evolution is to add new operations, and change small fractions of existing operations (fixing bugs and extending semantics).
However, such changes affect only new versions of operations from a new operation set, while old operations are used by specifying an appropriate `version`.
When an old `version` is specified, the behavior will be kept unchanged from that specified version to provide backward compatibility with older IRs.

A single ``xml`` file with IR may contain operations from different opsets.
An operation that is included in several opsets may be referred to with ``version`` which points to any opset that includes that operation.
For example, the same ``Convolution`` can be used with ``version="opset1"`` and ``version="opset2"`` because both opsets have the same ``Convolution`` operations.

