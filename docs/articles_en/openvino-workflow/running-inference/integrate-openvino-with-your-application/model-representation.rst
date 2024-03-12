.. {#openvino_docs_OV_UG_Model_Representation}

Model Representation in OpenVINO™ Runtime
===========================================


.. meta::
   :description: In OpenVINO™ Runtime a model is represented by special classes to work with model data types and shapes.


In OpenVINO™ Runtime, a model is represented by the ``ov::Model`` class.

The ``ov::Model`` object stores shared pointers to ``ov::op::v0::Parameter``, ``ov::op::v0::Result``, and ``ov::op::Sink`` operations,
which are inputs, outputs, and sinks of the graph. Sinks of the graph have no consumers and are not included in the results vector.
All other operations hold each other via shared pointers, in which a child operation holds its parent via a hard link. If an operation
has no consumers and is neither the ``Result`` nor the ``Sink`` operation whose shared pointer counter is zero, the operation will be
destructed and not be accessible anymore.

Each operation in ``ov::Model`` has the ``std::shared_ptr<ov::Node>`` type.

How OpenVINO Runtime Works with Models
#########################################

OpenVINO™ Runtime enables you to use different approaches to work with model inputs/outputs:

* The ``ov::Model::inputs()`` / ``ov::Model::outputs()``  methods are used to get vectors of all input/output ports.

* For a model that has only one input or output, you can use the ``ov::Model::input()`` or ``ov::Model::output()``  methods without
  any arguments to get input or output port respectively.

* The ``ov::Model::input()`` and ``ov::Model::output()``  methods can be used with the index of inputs or outputs from the framework
  model to get specific ports by index.

* You can use the tensor name of input or output from the original framework model together with the
  ``ov::Model::input()`` or ``ov::Model::output()`` methods to get specific ports. It means that you do not need to have any
  additional mapping of names from framework to OpenVINO as it was before. OpenVINO Runtime allows the usage of native framework
  tensor names, for example:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [all_inputs_ouputs]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [all_inputs_ouputs]


For details on how to build a model in OpenVINO™ Runtime, see the :ref:`Build a Model in OpenVINO Runtime <ov_ug_build_model>` section.

OpenVINO™ Runtime model representation uses special classes to work with model data types and shapes. The ``ov::element::Type``
is used for data types. See the section below for representation of shapes.

Representation of Shapes
###########################

OpenVINO™ Runtime provides two types for shape representation:

* ``ov::Shape`` - Represents static (fully defined) shapes.

* ``ov::PartialShape`` - Represents dynamic shapes. This means that the rank or some of dimensions are dynamic
  (dimension defines an interval or undefined).

``ov::PartialShape`` can be converted to ``ov::Shape`` by using the ``get_shape()`` method if all dimensions are static; otherwise,
the conversion will throw an exception. For example:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [ov:partial_shape]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:partial_shape]


However, in most cases, before getting static shape using the ``get_shape()`` method, you need to check if that shape is static.

Representation of Operations
################################

The ``ov::Op`` class represents any abstract operation in the model representation. Use this class to create
:doc:`custom operations <../../../documentation/openvino-extensibility/custom-openvino-operations>`.

Representation of Operation Sets
######################################

An operation set (opset) is a collection of operations that can be used to construct a model. The ``ov::OpSet`` class provides
the functionality to work with operation sets.
For each operation set, OpenVINO™ Runtime provides a separate namespace, for example ``opset8``.

Each OpenVINO™ Release introduces new operations and adds them to new operation sets, within which the new operations would change
the behavior of previous operations. Using operation sets helps you avoid changing your application when new operations are introduced.
For a complete list of operation sets supported in OpenVINO™ toolkit, see the :doc:`Available Operations Sets <../../../documentation/openvino-ir-format/operation-sets/available-opsets>`.
To add the support for custom operations, see :doc:`OpenVINO Extensibility Mechanism <../../../documentation/openvino-extensibility>`.

.. _ov_ug_build_model:

Building a Model in OpenVINO™ Runtime
###########################################

You can create a model from source. This section illustrates how to construct a model composed of operations from an available operation set.

Operation set ``opsetX`` integrates a list of pre-compiled operations that work for this purpose. In other words, ``opsetX``
defines a set of operations for building a graph.

To build an ``ov::Model`` instance from ``opset8`` operations, include the following files:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [import]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:include]


The following code demonstrates how to create a simple model:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [ov:create_simple_model]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:create_simple_model]


The following code creates a model with several outputs:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [ov:create_advanced_model]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:create_advanced_model]


Model Debugging Capabilities
###########################################

OpenVINO™ provides several debug capabilities:

* To receive additional messages about applied model modifications, rebuild the OpenVINO™ Runtime library with the
  ``-DENABLE_OPENVINO_DEBUG=ON`` option.

* Model can be visualized to image from the xDot format:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [ov:visualize]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:visualize]


.. code-block:: sh

   `ov::pass::VisualizeTree` can be parametrized via environment variables:

   OV_VISUALIZE_TREE_OUTPUT_SHAPES=1       - visualize shapes

   OV_VISUALIZE_TREE_OUTPUT_TYPES=1        - visualize types

   OV_VISUALIZE_TREE_MIN_MAX_DENORMAL=1    - pretty denormal values

   OV_VISUALIZE_TREE_RUNTIME_INFO=1        - print runtime information

   OV_VISUALIZE_TREE_IO=1                  - print I/O ports

   OV_VISUALIZE_TREE_MEMBERS_NAME=1        - print member names


* Also model can be serialized to IR:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [ov:serialize]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:serialize]


Additional Resources
########################

* :doc:`Available Operation Sets <../../../documentation/openvino-ir-format/operation-sets/available-opsets>`.
* :doc:`OpenVINO™ Runtime Extensibility Developer Guide <../../../documentation/openvino-extensibility>`.
* :doc:`Transformations Developer Guide <../../../documentation/openvino-extensibility/transformation-api>`.


