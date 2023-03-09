# OpenVINO Model Pass {#openvino_docs_Extensibility_UG_model_pass}

``:ref:`ov::pass::ModelPass <doxid-classov_1_1pass_1_1_model_pass>``` is used for transformations that take entire ``:ref:`ov::Model <doxid-classov_1_1_model>``` as an input and process it.

Template for ModelPass transformation class

.. doxygensnippet:: template_model_transformation.hpp
   :language: hpp
   :fragment: [model_pass:template_transformation_hpp]

.. doxygensnippet:: template_model_transformation.cpp
   :language: cpp
   :fragment: [model_pass:template_transformation_cpp]

Using ``:ref:`ov::pass::ModelPass <doxid-classov_1_1pass_1_1_model_pass>```, you need to override the ``run_on_model`` method where you will write the transformation code.
Return value is ``true`` if the original model has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is ``false``.
Also ``:ref:`ov::pass::ModelPass <doxid-classov_1_1pass_1_1_model_pass>``` based transformations can be executed via `ov::pass::Manager`.

See Also
########

* :doc:`OpenVINO™ Transformations <openvino_docs_transformations>`
