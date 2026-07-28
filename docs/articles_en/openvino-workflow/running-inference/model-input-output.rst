Model Input/Output
===============================================================================================

.. meta::
   :description:  OpenVINO™ Runtime includes several methods for handling model inputs and
                  outputs.


.. toctree::
   :maxdepth: 1
   :hidden:

   model-input-output/changing-input-shape
   model-input-output/dynamic-shapes
   model-input-output/string-tensors


``ov::Model::inputs()`` and  ``ov::Model::outputs()`` methods retrieve vectors of all
input/output ports.

Note that a similar logic is applied to retrieving data using the ``ov::InferRequest`` methods.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.py
         :language: cpp
         :fragment: [all_inputs_ouputs]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
         :language: cpp
         :fragment: [all_inputs_ouputs]


``ov::Model::input()`` and ``ov::Model::output()`` methods retrieve vectors of specific
input/output ports. To select the ports, you may use:

* no arguments, if the model has only one input or output,
* the index of inputs or outputs from the original model framework,

  .. tab-set::

     .. tab-item:: Python
        :sync: py

        .. code-block:: python

            ov_model_input = model.input(index)
            ov_model_output = model.output(index)

     .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

           auto ov_model_input = ov_model->input(index);
           auto ov_model_output = ov_model->output(index);


* tensor names of inputs or outputs from the original model framework.

  .. tab-set::

     .. tab-item:: Python
        :sync: py

        .. code-block:: python

           ov_model_input = model.input(original_fw_in_tensor_name)
           ov_model_output = model.output(original_fw_out_tensor_name)

     .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

           auto ov_model_input = ov_model->input(original_fw_in_tensor_name);
           auto ov_model_output = ov_model->output(original_fw_out_tensor_name);


Since all ``ov::Model`` inputs and outputs are always numbered, using the index is the
recommended way. That is because the original frameworks do not necessarily require tensor
names, and so, ``ov::Model`` may contain an empty list of tensor_names for inputs/outputs.
The ``get_any_name`` and ``get_names`` methods enable you to retrieve one or all tensor names
associated with an input/output. If the names are not present, the methods will
return empty names.

For information on how ``ov::InferRequest`` methods retrieve vectors of input output ports,
see the :doc:`Inference Request <inference-request>` article.

For more details on how to work with model inputs and outputs, see other articles in this category:

- :doc:`Changing Input Shapes <model-input-output/changing-input-shape>`
- :doc:`Dynamic Shape Models <model-input-output/dynamic-shapes>`
- :doc:`String Tensors <model-input-output/string-tensors>`
