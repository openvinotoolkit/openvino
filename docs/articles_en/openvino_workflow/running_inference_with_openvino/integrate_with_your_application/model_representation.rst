.. {#openvino_docs_OV_UG_Model_Representation}

Model Representation in OpenVINO™ Runtime
===========================================


.. meta::
   :description: In OpenVINO™ Runtime a model is represented by special classes to work with model data types and shapes.


In OpenVINO™ Runtime, a model is represented by the ``:ref:`ov::Model <doxid-classov_1_1_model>```  class.

The ``:ref:`ov::Model <doxid-classov_1_1_model>``` object stores shared pointers to ``:ref:`ov::op::v0::Parameter <doxid-classov_1_1op_1_1v0_1_1_parameter>```, ``:ref:`ov::op::v0::Result <doxid-classov_1_1op_1_1v0_1_1_result>```, and ``:ref:`ov::op::Sink <doxid-classov_1_1op_1_1_sink>``` operations, which are inputs, outputs, and sinks of the graph. Sinks of the graph have no consumers and are not included in the results vector. All other operations hold each other via shared pointers, in which a child operation holds its parent via a hard link. If an operation has no consumers and is neither the ``Result`` nor the ``Sink`` operation whose shared pointer counter is zero, the operation will be destructed and not be accessible anymore.

Each operation in ``:ref:`ov::Model <doxid-classov_1_1_model>``` has the ``std::shared_ptr<:ref:`ov::Node <doxid-classov_1_1_node>`>`` type.

How OpenVINO Runtime Works with Models
#########################################

OpenVINO™ Runtime enables you to use different approaches to work with model inputs/outputs:

* The ``:ref:`ov::Model::inputs() <doxid-classov_1_1_model_1ac28a4c66071e165c4f98906ab489e5d5>``` / ``:ref:`ov::Model::outputs() <doxid-classov_1_1_model_1af6e381008712ce22d6f4b93b87303dd8>```  methods are used to get vectors of all input/output ports.

* For a model that has only one input or output, you can use the ``:ref:`ov::Model::input() <doxid-classov_1_1_model_1a5deeced6688795bc6cdad9ce74d972e7>``` or ``:ref:`ov::Model::output() <doxid-classov_1_1_model_1a54c76c98bc7dd8fb04e866d06134efc7>```  methods without any arguments to get input or output port respectively.

* The ``:ref:`ov::Model::input() <doxid-classov_1_1_model_1a5deeced6688795bc6cdad9ce74d972e7>``` and ``:ref:`ov::Model::output() <doxid-classov_1_1_model_1a54c76c98bc7dd8fb04e866d06134efc7>```  methods can be used with the index of inputs or outputs from the framework model to get specific ports by index.

* You can use the tensor name of input or output from the original framework model together with the ``:ref:`ov::Model::input() <doxid-classov_1_1_model_1a5deeced6688795bc6cdad9ce74d972e7>``` or ``:ref:`ov::Model::output() <doxid-classov_1_1_model_1a54c76c98bc7dd8fb04e866d06134efc7>``` methods to get specific ports. It means that you do not need to have any additional mapping of names from framework to OpenVINO as it was before. OpenVINO Runtime allows the usage of native framework tensor names, for example:

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


For details on how to build a model in OpenVINO™ Runtime, see the :ref:`Build a Model in OpenVINO Runtime <ov_ug_build_model>`  section.

OpenVINO™ Runtime model representation uses special classes to work with model data types and shapes. The ``:ref:`ov::element::Type <doxid-classov_1_1element_1_1_type>```  is used for data types. See the section below for representation of shapes.

Representation of Shapes
###########################

OpenVINO™ Runtime provides two types for shape representation: 

* ``:ref:`ov::Shape <doxid-classov_1_1_shape>``` - Represents static (fully defined) shapes.

* ``:ref:`ov::PartialShape <doxid-classov_1_1_partial_shape>``` - Represents dynamic shapes. This means that the rank or some of dimensions are dynamic (dimension defines an interval or undefined). 

``:ref:`ov::PartialShape <doxid-classov_1_1_partial_shape>``` can be converted to ``:ref:`ov::Shape <doxid-classov_1_1_shape>``` by using the ``get_shape()`` method if all dimensions are static; otherwise, the conversion will throw an exception. For example: 

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

The ``ov::Op`` class represents any abstract operation in the model representation. Use this class to create :doc:`custom operations <openvino_docs_Extensibility_UG_add_openvino_ops>`.

Representation of Operation Sets
######################################

An operation set (opset) is a collection of operations that can be used to construct a model. The ``:ref:`ov::OpSet <doxid-classov_1_1_op_set>``` class provides the functionality to work with operation sets.
For each operation set, OpenVINO™ Runtime provides a separate namespace, for example ``opset8``.

Each OpenVINO™ Release introduces new operations and adds them to new operation sets, within which the new operations would change the behavior of previous operations. Using operation sets helps you avoid changing your application when new operations are introduced.
For a complete list of operation sets supported in OpenVINO™ toolkit, see the :doc:`Available Operations Sets <openvino_docs_ops_opset>`.
To add the support for custom operations, see :doc:`OpenVINO Extensibility Mechanism <openvino_docs_Extensibility_UG_Intro>`.

.. _ov_ug_build_model:

Building a Model in OpenVINO™ Runtime
###########################################

You can create a model from source. This section illustrates how to construct a model composed of operations from an available operation set.

Operation set ``opsetX`` integrates a list of pre-compiled operations that work for this purpose. In other words, ``opsetX`` defines a set of operations for building a graph.

To build an ``:ref:`ov::Model <doxid-classov_1_1_model>``` instance from ``opset8`` operations, include the following files:

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

* To receive additional messages about applied model modifications, rebuild the OpenVINO™ Runtime library with the ``-DENABLE_OPENVINO_DEBUG=ON`` option.

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

* :doc:`Available Operation Sets <openvino_docs_ops_opset>`.
* :doc:`OpenVINO™ Runtime Extensibility Developer Guide <openvino_docs_Extensibility_UG_Intro>`.
* :doc:`Transformations Developer Guide <openvino_docs_transformations>`.


