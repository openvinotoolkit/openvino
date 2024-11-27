OpenVINO Model Pass
===================


.. meta::
   :description: Learn how to use Model Pass transformation class to take entire
                 ov::Model as input and process it.


``ov::pass::ModelPass`` is used for transformations that take entire ``ov::Model`` as an input and process it.

Template for ModelPass transformation class

.. doxygensnippet:: docs/articles_en/assets/snippets/template_model_transformation.hpp
   :language: cpp
   :fragment: [model_pass:template_transformation_hpp]

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/template_model_transformation.cpp
         :language: cpp
         :fragment: [model_pass:template_transformation_cpp]

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_pass.py
         :language: py
         :fragment: [model_pass:ov_model_pass_py]

Using ``ov::pass::ModelPass``, you need to override the ``run_on_model`` method where you will write the transformation code.
Return value is ``true`` if the original model has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is ``false``.
Also ``ov::pass::ModelPass`` based transformations can be executed via ``ov::pass::Manager``.

See Also
########

* :doc:`OpenVINO™ Transformations <../transformation-api>`

