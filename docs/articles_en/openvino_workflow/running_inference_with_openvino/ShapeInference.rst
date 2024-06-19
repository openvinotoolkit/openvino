.. {#openvino_docs_OV_UG_ShapeInference}

Changing Input Shapes
=====================

.. meta::
   :description: OpenVINO™ allows changing model input shape during the runtime when the provided
                 input has a different size than the model's input shape.


OpenVINO™ enables you to change model input shape during the application runtime.
It may be useful when you want to feed the model an input that has different size than the model input shape.
The following instructions are for cases where you need to change the model input shape repeatedly.

.. note::

   If you need to do this only once, prepare a model with updated shapes via
   :doc:`Model Conversion API <openvino_docs_model_processing_introduction>`.
   For more information, refer to the :doc:`Setting Input Shapes <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Converting_Model>` article.


The reshape method
########################

The reshape method is used as ``ov::Model::reshape`` in C++ and
`Model.reshape <api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape>`__
in Python. The method updates input shapes and propagates them down to the outputs
of the model through all intermediate layers. The code below is an example of how
to set a new batch size with the ``reshape`` method:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ShapeInference.py
         :language: Python
         :fragment: picture_snippet

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ShapeInference.cpp
         :language: cpp
         :fragment: picture_snippet

The diagram below presents the results of using the method, where the size of
model input is changed with an image input:

.. image:: _static/images/original_vs_reshaped_model.svg

When using the ``reshape`` method, you may take one of the approaches:

.. _usage_of_reshape_method:


1. You can pass a new shape to the method in order to change the input shape of
   the model with a single input. See the example of adjusting spatial dimensions to the input image:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/snippets/ShapeInference.py
            :language: python
            :fragment: simple_spatials_change

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/snippets/ShapeInference.cpp
            :language: cpp
            :fragment: spatial_reshape


   To do the opposite - to resize input image to match the input shapes of the model,
   use the :doc:`pre-processing API <openvino_docs_OV_UG_Preprocessing_Overview>`.


2. You can express a reshape plan, specifying the input by the port, the index, and the tensor name:

   .. tab-set::

      .. tab-item:: Port

         .. tab-set::

            .. tab-item:: Python
               :sync: py

               ``openvino.runtime.Output`` dictionary key specifies input by passing actual input object.
               Dictionary values representing new shapes could be ``PartialShape``:

               .. doxygensnippet:: docs/snippets/ShapeInference.py
                  :language: python
                  :fragment: [obj_to_shape]

            .. tab-item:: C++
               :sync: cpp

               ``map<ov::Output<ov::Node>, ov::PartialShape`` specifies input by passing actual input port:

               .. doxygensnippet:: docs/snippets/ShapeInference.cpp
                  :language: cpp
                  :fragment: [obj_to_shape]

      .. tab-item:: Index

         .. tab-set::

            .. tab-item:: Python
               :sync: py

               ``int`` dictionary key specifies input by its index.
               Dictionary values representing new shapes could be ``tuple``:

               .. doxygensnippet:: docs/snippets/ShapeInference.py
                  :language: python
                  :fragment: [idx_to_shape]

            .. tab-item:: C++
               :sync: cpp

               ``map<size_t, ov::PartialShape>`` specifies input by its index:

               .. doxygensnippet:: docs/snippets/ShapeInference.cpp
                  :language: cpp
                  :fragment: [idx_to_shape]

      .. tab-item:: Tensor Name

         .. tab-set::

            .. tab-item:: Python
               :sync: py

               ``str`` dictionary key specifies input by its name.
               Dictionary values representing new shapes could be ``str``:

               .. doxygensnippet:: docs/snippets/ShapeInference.py
                  :language: python
                  :fragment: [name_to_shape]

            .. tab-item:: C++
               :sync: cpp

               ``map<string, ov::PartialShape>`` specifies input by its name:

               .. doxygensnippet:: docs/snippets/ShapeInference.cpp
                  :language: cpp
                  :fragment: [name_to_shape]


You can find the usage scenarios of the ``reshape`` method in
:doc:`Hello Reshape SSD Samples <openvino_sample_hello_reshape_ssd>`.

.. note::

   In some cases, models may not be ready to be reshaped. Therefore, a new input
   shape cannot be set neither with :doc:`Model Conversion API <openvino_docs_model_processing_introduction>`
   nor the ``reshape`` method.

The set_batch method
########################

The meaning of the model batch may vary depending on the model design.
To change the batch dimension of the model, :ref:`set the layout <declare_model_s_layout>` and call the ``set_batch`` method.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ShapeInference.py
         :language: Python
         :fragment: set_batch

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ShapeInference.cpp
         :language: cpp
         :fragment: set_batch


The ``set_batch`` method is a high-level API of the reshape functionality, so all
information about the ``reshape`` method implications are applicable for ``set_batch`` too.

Once you set the input shape of the model, call the ``compile_model`` method to
get a ``CompiledModel`` object for inference with updated shapes.

There are other approaches to change model input shapes during the stage of
:ref:`IR generation <when_to_specify_input_shapes>` or :ref:`model representation <openvino_docs_OV_UG_Model_Representation>` in OpenVINO Runtime.


.. important::

   Shape-changing functionality could be used to turn dynamic model input into a
   static one and vice versa. Always set static shapes when the shape of data is
   NOT going to change from one inference to another. Setting static shapes can
   avoid memory and runtime overheads for dynamic shapes which may vary depending
   on hardware plugin and model used. For more information, refer to the
   :doc:`Dynamic Shapes <openvino_docs_OV_UG_DynamicShapes>`.


Additional Resources
####################

* :doc:`Extensibility documentation <openvino_docs_Extensibility_UG_Intro>` - describes a special mechanism in OpenVINO that allows adding support of shape inference for custom operations.
* `ov::Model::reshape <classov_1_1Model.html#doxid-classov-1-1-model-1aa21aff80598d5089d591888a4c7f33ae>`__ - in OpenVINO Runtime C++ API
* `Model.reshape <api/ie_python_api/_autosummary/openvino.runtime.Model.html#openvino.runtime.Model.reshape>`__ - in OpenVINO Runtime Python API.
* :doc:`Dynamic Shapes <openvino_docs_OV_UG_DynamicShapes>`
* :doc:`OpenVINO samples <openvino_docs_OV_UG_Samples_Overview>`
* :doc:`Preprocessing API <openvino_docs_OV_UG_Preprocessing_Overview>`

