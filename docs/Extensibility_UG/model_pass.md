# OpenVINO Model Pass {#openvino_docs_Extensibility_UG_model_pass}

@sphinxdirective

.. meta::
   :description: Learn how to use Model Pass transformation class to take entire 
                 ov::Model as input and process it.


``:ref:`ov::pass::ModelPass <doxid-classov_1_1pass_1_1_model_pass>``` is used for transformations that take entire ``:ref:`ov::Model <doxid-classov_1_1_model>``` as an input and process it.

Template for ModelPass transformation class

.. doxygensnippet:: docs/snippets/template_model_transformation.hpp 
   :language: cpp 
   :fragment: [model_pass:template_transformation_hpp]

.. doxygensnippet:: docs/snippets/template_model_transformation.cpp
   :language: cpp
   :fragment: [model_pass:template_transformation_cpp]

Using ``:ref:`ov::pass::ModelPass <doxid-classov_1_1pass_1_1_model_pass>```, you need to override the ``run_on_model`` method where you will write the transformation code.
Return value is ``true`` if the original model has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is ``false``.
Also ``:ref:`ov::pass::ModelPass <doxid-classov_1_1pass_1_1_model_pass>``` based transformations can be executed via ``:ref:`ov::pass::Manager <doxid-classov_1_1pass_1_1_manager>```.

See Also
########

* :doc:`OpenVINO™ Transformations <openvino_docs_transformations>`

@endsphinxdirective
