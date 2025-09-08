Model Representation in OpenVINO™ Runtime
===============================================================================================


.. meta::
   :description: In OpenVINO™ Runtime a model is represented by special classes to work with
                 model data types and shapes.


The model is represented by the ``ov::Model class``.
The ``ov::Model`` object stores shared pointers to the following operations:

* ``ov::op::v0::Parameter`` - inputs
* ``ov::op::v0::Result`` - outputs
* ``ov::op::Sink`` - sinks (have no consumers and are not included in the results vector)

Other operations hold each other via shared pointers, where a child operation points to its
parent via a hard link. If an operation has no consumers and is neither a ``Result`` nor a
``Sink`` operation whose shared pointer counter is zero, the operation will be removed.
Each operation in ``ov::Model`` has the ``std::shared_ptr<ov::Node>`` type.

Classes and methods use to work with models
###############################################################################################

Data types
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The ``ov::element::Type`` is used as data type representation.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: python

         ov_input.get_element_type()

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         ov_input->get_element_type();


Shapes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The OpenVINO runtime provides two types of shape representation:

* ``ov::Shape`` (read more) - represents static (fully defined) shapes,
* ``ov::PartialShape`` (read more) - represents shapes that may be partially or fully
  dynamic (with a given or an undefined rank).

``ov::PartialShape`` can be converted to ``ov::Shape``, but only if all dimensions are static.
Do this with the ``get_shape()`` method.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
         :language: python
         :fragment: [ov:partial_shape]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:partial_shape]


Keep in mind that OpenVINO enables you to change the model input shape during the application
runtime. It also supports models accepting dynamic shapes. For more details, see
:doc:`Model Input/Output Handling <model-input-output>`.


Operations and Operation Sets
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Each abstract operation in a model is represented by the ``ov::Op class``.
A collection of operations used to construct a model is represented by the ``ov::OpSet``
class. It refers to an operation set defined by an individual **opsetX** namespace,
for example **opset15**. These opsets are added with new OpenVINO releases to change the
behavior of previously used operations and help you avoid changing your application when new
operations are introduced. For more information, check the documentation on
:doc:`OpenVINO model <../../documentation/openvino-ir-format>` and
:doc:`opsets <../../documentation/openvino-ir-format/operation-sets>`, as well as
:doc:`custom operations <../../documentation/openvino-extensibility>`.


Build a Model in OpenVINO™ Runtime
###############################################################################################

You can create a model from source by constructing it with operations from an available
**opsetX** operation set. Each operation set integrates a list of pre-compiled operations
that work for this purpose. In other words, **opsetX** defines a set of operations for
building a model. To create an ``ov::Model`` instance from ``opset15`` operations, include
the following libraries:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [import]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [ov:include]


Then, you can create a model, as shown in these two examples:

.. tab-set::

   .. tab-item:: Simple (single output)

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
               :language: cpp
               :fragment: [ov:create_simple_model]

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
               :language: cpp
               :fragment: [ov:create_simple_model]

   .. tab-item:: More advanced (multiple outputs)

      .. tab-set::

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
               :language: cpp
               :fragment: [ov:create_advanced_model]

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
               :language: cpp
               :fragment: [ov:create_advanced_model]


Debug a Model
###############################################################################################

Here is a list of features that can help you debug models in OpenVINO:

* ``DENABLE_OPENVINO_DEBUG=ON`` - used to rebuild the OpenVINO Runtime library, to receive
  additional messages about applied model modifications
* ``ov::pass::VisualizeTree`` - used to save ``ov::Model`` to xDot format. It can be
  parametrized via environment variables:

  * ``OV_VISUALIZE_TREE_OUTPUT_SHAPES=1`` - visualize shapes
  * ``OV_VISUALIZE_TREE_OUTPUT_TYPES=1`` - visualize types
  * ``OV_VISUALIZE_TREE_MIN_MAX_DENORMAL=1`` - pretty denormal values
  * ``OV_VISUALIZE_TREE_RUNTIME_INFO=1`` - print runtime information
  * ``OV_VISUALIZE_TREE_IO=1`` - print I/O ports
  * ``OV_VISUALIZE_TREE_MEMBERS_NAME=1`` - print member names

* ``ov::serialize`` - used to save a model to IR:

  .. tab-set::

     .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
           :language: python
           :fragment: [ov:serialize]

     .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
           :language: cpp
           :fragment: [ov:serialize]


  .. note::

     While ``save_model`` is the preferred method for saving a model to IR, it is
     recommended to use ``serialize`` for debugging purposes. It requires less time and
     computational resources, as it does not apply weight compression.


* **xDot format to image conversion** - used to visualize model graphs:

  .. tab-set::

     .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
           :language: python
           :fragment: [ov:visualize]

     .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
           :language: cpp
           :fragment: [ov:visualize]


Additional Resources
########################

* :doc:`Available Operation Sets <../../../documentation/openvino-ir-format/operation-sets/available-opsets>`.
* :doc:`OpenVINO™ Runtime Extensibility Developer Guide <../../../documentation/openvino-extensibility>`.
* :doc:`Transformations Developer Guide <../../../documentation/openvino-extensibility/transformation-api>`.
