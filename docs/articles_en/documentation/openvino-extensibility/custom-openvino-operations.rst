Custom OpenVINO Operations
=============================


.. meta::
   :description: Explore OpenVINO™ Extension API which enables registering
                 custom operations to support models with operations
                 not supported by OpenVINO.

OpenVINO™ Extension API allows you to register custom operations to support models with operations which OpenVINO™ does not support out-of-the-box. This capability requires writing code in C++, so if you are using Python to develop your application you need to build a separate shared library implemented in C++ first and load it in Python using ``add_extension`` API. Please refer to :ref:`Create library with extensions <create_a_library_with_extensions>` for more details on library creation and usage. The remaining part of this document describes how to implement an operation class.

Operation Class
###############

To add your custom operation, create a new class that extends ``ov::Op``, which is in turn derived from ``ov::Node``, the base class for all graph operations in OpenVINO™. To add ``ov::Op``, include the next file:

.. doxygensnippet:: src/core/template_extension/identity.hpp
   :language: cpp
   :fragment: [op:common_include]

Follow the steps below to add a custom operation:

1. Add the ``OPENVINO_OP`` macro which defines a ``NodeTypeInfo`` object that identifies the type of the operation to the graph users and helps with dynamic type resolution. The type info of an operation currently consists of a string operation identifier and a string for operation version.

2. Implement default constructor and constructors that optionally take the operation inputs and attributes as parameters.

3. Override the shape inference method ``validate_and_infer_types``. This method is called multiple times during graph manipulations to determine the shapes and element types of the operations outputs. To access the input shapes and input element types, use the ``get_input_partial_shape()`` and ``get_input_element_type()`` methods of ``ov::Node``. Set the inferred shape and element type of the output using ``set_output_type``.

4. Override the ``clone_with_new_inputs`` method, which enables graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.

5. Override the ``visit_attributes`` method, which enables serialization and deserialization of operation attributes. An ``AttributeVisitor`` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware ``on_attribute`` helper. Helpers are already implemented for standard C++ types like ``int64_t``, ``float``, ``bool``, ``vector``, and for existing OpenVINO defined types.

6. Override ``evaluate``, which is an optional method that enables fallback of some devices to this implementation and the application of constant folding if there is a custom operation on the constant branch. If your operation contains ``evaluate`` method you also need to override the ``has_evaluate`` method, this method allows to get information about availability of ``evaluate`` method for the operation.

Based on that, declaration of an operation class can look as follows:


Operation Constructors
++++++++++++++++++++++

OpenVINO™ operation contains two constructors:

* Default constructor, which enables you to create an operation without attributes
* Constructor that creates and validates an operation with specified inputs and attributes

.. doxygensnippet:: src/core/template_extension/identity.cpp
   :language: cpp
   :fragment: [op:ctor]

``validate_and_infer_types()``
++++++++++++++++++++++++++++++

``ov::Node::validate_and_infer_types`` method validates operation attributes and calculates output shapes using attributes of the operation.

.. doxygensnippet:: src/core/template_extension/identity.cpp
   :language: cpp
   :fragment: [op:validate]

``clone_with_new_inputs()``
+++++++++++++++++++++++++++

``ov::Node::clone_with_new_inputs`` method creates a copy of the operation with new inputs.

.. doxygensnippet:: src/core/template_extension/identity.cpp
   :language: cpp
   :fragment: [op:copy]

``visit_attributes()``
++++++++++++++++++++++

``ov::Node::visit_attributes`` method enables you to visit all operation attributes.

.. doxygensnippet:: src/core/template_extension/identity.cpp
   :language: cpp
   :fragment: [op:visit_attributes]

``evaluate() and has_evaluate()``
+++++++++++++++++++++++++++++++++

``ov::Node::evaluate`` method enables you to apply constant folding to an operation.

.. doxygensnippet:: src/core/template_extension/identity.cpp
   :language: cpp
   :fragment: [op:evaluate]

