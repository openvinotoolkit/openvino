.. {#openvino_docs_Extensibility_UG_model_pass}

OpenVINO Model Pass
===================


.. meta::
   :description: Learn how to use Model Pass transformation class to take entire
                 ov::Model as input and process it.


``ov::pass::ModelPass`` is used for transformations that take entire ``ov::Model`` as an input and process it.

To create your own transformation (based on ModelPass) you need to:

- Create own class inherited from `ov::pass::ModelPass``
- Add OPENVINO_RTTI line. This macro puts necessary declarations of RTTI block inside a class definition.
  First argument is class name, second is version.
- Override ``run_on_model`` virtual method from parent class

Template for ModelPass transformation class

.. doxygensnippet:: docs/snippets/template_model_transformation.hpp
   :language: cpp
   :fragment: [model_pass:template_transformation_hpp]

.. doxygensnippet:: docs/snippets/template_model_transformation.cpp
   :language: cpp
   :fragment: [model_pass:template_transformation_cpp]

Using ``ov::pass::ModelPass``, you need to override the ``run_on_model`` method where you will write the transformation code.
Return value is ``true`` if the original model has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is ``false``.
Also ``ov::pass::ModelPass`` based transformations can be executed via ``ov::pass::Manager``.

.. doxygensnippet:: docs/snippets/model-pass.cpp
   :language: cpp
   :fragment: [model_pass:model_pass_cpp]

This code will run ``ov::pass::MyModelTransformation::run_on_model`` passing model as argument to it.
In real applications there might be a lot of such transformations registered in ``ov::pass::Manager``.
All of these transformations will be executed one after another in order they were registered.

See Also
########

* :doc:`OpenVINO™ Transformations <../transformation-api>`

