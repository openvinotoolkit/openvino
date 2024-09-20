[LEGACY] Graph Traversal and Modification
===========================================

.. meta::
   :description: Learn about deprecated APIs and the Port and Connection classes
                 in Model Optimizer used for graph traversal and transformation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated TensorFlow conversion method. The guide on the new and recommended method, using a new frontend, can be found in the  :doc:`Frontend Extensions <../../../openvino-extensibility/frontend-extensions>` article.

There are three APIs for a graph traversal and transformation used in the Model Optimizer:

1. The API provided with the ``networkx`` Python library for the ``networkx.MultiDiGraph`` class, which is the base class for
the ``mo.graph.graph.Graph`` object. For example, the following methods belong to this API level:

* ``graph.add_edges_from([list])``,
* ``graph.add_node(x, attrs)``,
* ``graph.out_edges(node_id)``
* other methods where ``graph`` is a an instance of the ``networkx.MultiDiGraph`` class.

**This is the lowest-level API. Avoid using it in the Model Optimizer transformations**. For more details, refer to the :ref:`Model Representation in Memory <mo_model_representation_in_memory>` section.

2. The API built around the ``mo.graph.graph.Node`` class. The ``Node`` class is the primary class to work with graph nodes
and their attributes. Examples of such methods and functions are:

* ``node.in_node(y)``,
* ``node.out_node(x)``,
* ``node.get_outputs()``,
* ``node.insert_node_after(n1, y)``,
* ``create_edge(n1, n2)``

**There are some "Node" class methods not recommended for use and some functions defined in the mo.graph.graph have been deprecated**. For more details, refer to the ``mo/graph/graph.py`` file.

3. The high-level API called Model Optimizer Graph API, which uses ``mo.graph.graph.Graph``, ``mo.graph.port.Port`` and
``mo.graph.connection.Connection`` classes. For example, the following methods belong to this API level:

* ``node.in_port(x)``,
* ``node.out_port(y)``,
* ``port.get_connection()``,
* ``connection.get_source()``,
* ``connection.set_destination(dest_port)``

**This is the recommended API for the Model Optimizer transformations and operations implementation**.

The main benefit of using the Model Optimizer Graph API is that it hides some internal implementation details (the fact that
the graph contains data nodes), provides API to perform safe and predictable graph manipulations, and adds operation
semantic to the graph. This is achieved with introduction of concepts of ports and connections.

.. note::
   This article is dedicated to the Model Optimizer Graph API only and does not cover other two non-recommended APIs.

.. _mo_intro_ports:

=====
Ports
=====

An operation semantic describes how many inputs and outputs the operation has. For example,
:doc:`Parameter <../../../openvino-ir-format/operation-sets/operation-specs/infrastructure/parameter-1>` and :doc:`Const <../../../openvino-ir-format/operation-sets/operation-specs/infrastructure/constant-1>` operations have no
inputs and have one output, :doc:`ReLU <../../../openvino-ir-format/operation-sets/operation-specs/activation/relu-1>` operation has one input and one output,
:doc:`Split <../../../openvino-ir-format/operation-sets/operation-specs/movement/split-1>` operation has 2 inputs and a variable number of outputs depending on the value of the
attribute ``num_splits``.

Each operation node in the graph (an instance of the ``Node`` class) has 0 or more input and output ports (instances of
the ``mo.graph.port.Port`` class). The ``Port`` object has several attributes:

* ``node`` - the instance of the ``Node`` object the port belongs to.
* ``idx`` - the port number. Input and output ports are numbered independently, starting from ``0``. Thus,
  :doc:`ReLU <../../../openvino-ir-format/operation-sets/operation-specs/activation/relu-1>` operation has one input port (with index ``0``) and one output port (with index ``0``).
* ``type`` - the type of the port. Could be equal to either ``"in"`` or ``"out"``.
* ``data`` - the object that should be used to get attributes of the corresponding data node. This object has methods ``get_shape()`` / ``set_shape()`` and ``get_value()`` / ``set_value()`` to get/set shape/value of the corresponding data node. For example, ``in_port.data.get_shape()`` returns an input shape of a tensor connected to input port ``in_port`` (``in_port.type == 'in'``), ``out_port.data.get_value()`` returns a value of a tensor produced from output port ``out_port`` (``out_port.type == 'out'``).

.. note::
   Functions ``get_shape()`` and ``get_value()`` return ``None`` until the partial inference phase. For more information  about model conversion phases, refer to the :ref:`Model Conversion Pipeline <mo_model_conversion_pipeline>`. For information about partial inference phase, see the :ref:`Partial Inference <mo_partial_inference>`.

There are several methods of the ``Node`` class to get the instance of a corresponding port:

* ``in_port(x)`` and ``out_port(x)`` to get the input/output port with number ``x``.
* ``in_ports()`` and ``out_ports()`` to get a dictionary, where key is a port number and the value is the corresponding input/output port.

Attributes ``in_ports_count`` and ``out_ports_count`` of the ``Op`` class instance define default number of input and output
ports to be created for the ``Node``. However, additional input/output ports can be added using methods
``add_input_port()`` and ``add_output_port()``. Port also can be removed, using the ``delete_input_port()`` and
``delete_output_port()`` methods.

The ``Port`` class is just an abstraction that works with edges incoming/outgoing to/from a specific ``Node`` instance. For
example, output port with ``idx = 1`` corresponds to the outgoing edge of a node with an attribute ``out = 1``, the input
port with ``idx = 2`` corresponds to the incoming edge of a node with an attribute ``in = 2``.

Consider the example of a graph part with 4 operation nodes "Op1", "Op2", "Op3", and "Op4" and a number of data nodes
depicted with light green boxes.

.. image:: ../../../../assets/images/MO_ports_example_1.svg
   :scale: 80 %
   :align: center

Operation nodes have input ports (yellow squares) and output ports (light purple squares). Input port may not be
connected. For example, the input **port 2** of node **Op1** does not have incoming edge, while output port always has an
associated data node (after the partial inference when the data nodes are added to the graph), which may have no
consumers.

Ports can be used to traverse a graph. The method ``get_source()`` of an input port returns an output port producing the
tensor consumed by the input port. It is important that the method works the same during front, middle and back phases of a
model conversion even though the graph structure changes (there are no data nodes in the graph during the front phase).

Let's assume that there are 4 instances of ``Node`` object ``op1, op2, op3``, and ``op4`` corresponding to nodes **Op1**, **Op2**,
**Op3**, and **Op4**, respectively. The result of ``op2.in_port(0).get_source()`` and ``op4.in_port(1).get_source()`` is the
same object ``op1.out_port(1)`` of type ``Port``.

The method ``get_destination()`` of an output port returns the input port of the node consuming this tensor. If there are
multiple consumers of this tensor, the error is raised. The method ``get_destinations()`` of an output port returns a
list of input ports consuming the tensor.

The method ``disconnect()`` removes a node incoming edge corresponding to the specific input port. The method removes
several edges if it is applied during the front phase for a node output port connected with multiple nodes.

The method ``port.connect(another_port)`` connects output port ``port`` and input port ``another_port``. The method handles
situations when the graph contains data nodes (middle and back phases) and does not create an edge between two nodes
but also automatically creates data node or reuses existing data node. If the method is used during the front phase and
data nodes do not exist, the method creates edge and properly sets ``in`` and ``out`` edge attributes.

For example, applying the following two methods to the graph above will result in the graph depicted below:

.. code-block:: py
   :force:

   op4.in_port(1).disconnect()
   op3.out_port(0).connect(op4.in_port(1))

.. image:: ../../../../assets/images/MO_ports_example_2.svg
   :scale: 80 %
   :align: center

.. note::
   For a full list of available methods, refer to the ``Node`` class implementation in the ``mo/graph/graph.py`` and ``Port`` class implementation in the ``mo/graph/port.py`` files.

===========
Connections
===========

Connection is a concept introduced to easily and reliably perform graph modifications. Connection corresponds to a
link between a source output port with one or more destination input ports or a link between a destination input port
and source output port producing data. So each port is connected with one or more ports with help of a connection.
Model Optimizer uses the ``mo.graph.connection.Connection`` class to represent a connection.

There is only one ``get_connection()`` method of the ``Port`` class to get the instance of the corresponding ``Connection``
object. If the port is not connected, the returned value is ``None``.

For example, the ``op3.out_port(0).get_connection()`` method returns a ``Connection`` object encapsulating edges from node
**Op3** to data node **data_3_0** and two edges from data node **data_3_0** to two ports of the node **Op4**.

The ``Connection`` class provides methods to get source and destination(s) ports the connection corresponds to:

* ``connection.get_source()`` - returns an output ``Port`` object producing the tensor.
* ``connection.get_destinations()`` - returns a list of input ``Port`` consuming the data.
* ``connection.get_destination()`` - returns a single input ``Port`` consuming the data. If there are multiple consumers, the exception is raised.

The ``Connection`` class provides methods to modify a graph by changing a source or destination(s) of a connection. For
example, the function call ``op3.out_port(0).get_connection().set_source(op1.out_port(0))`` changes source port of edges
consuming data from port ``op3.out_port(0)`` to ``op1.out_port(0)``. The transformed graph from the sample above is depicted
below:

.. image:: ../../../../assets/images/MO_connection_example_1.svg
   :scale: 80 %
   :align: center

Another example is the ``connection.set_destination(dest_port)`` method. It disconnects ``dest_port`` and all input ports to which
the connection is currently connected and connects the connection source port to ``dest_port``.

Note that connection works seamlessly during front, middle, and back phases and hides the fact that the graph structure is
different.

.. note::
   For a full list of available methods, refer to the ``Connection`` class implementation in the ``mo/graph/connection.py`` file.

====================
Additional Resources
====================

* :doc:`Model Optimizer Extensibility <../legacy-model-optimizer-extensibility>`
* :doc:`Model Optimizer Extensions <[legacy]-model-optimizer-extensions>`
* :doc:`Extending Model Optimizer with Caffe Python Layers <[legacy]-extending-model-optimizer-with-caffe-python-layers>`

