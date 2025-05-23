Use Case - Integrate and Save Preprocessing Steps Into IR
=========================================================


.. meta::
   :description: Once a model is read, the preprocessing/ postprocessing steps
                 can be added and then the resulting model can be saved to
                 OpenVINO Intermediate Representation.


Previous sections covered the :doc:`preprocessing steps <../preprocessing-api-details>`
and the overview of :doc:`Layout API <../layout-api-overview>`.

For many applications, it is also important to minimize read/load time of a model.
Therefore, performing integration of preprocessing steps every time on application
startup, after ``ov::runtime::Core::read_model``, may seem inconvenient. In such cases,
once pre and postprocessing steps have been added, it can be useful to store new execution
model to OpenVINO Intermediate Representation (OpenVINO IR, `.xml` format).

Most available preprocessing steps can also be performed via command-line options,
using ``ovc``. For details on such command-line options, refer to the
:ref:`Model Conversion <convert_model_cli_ovc>`.

Code example - Saving Model with Preprocessing to OpenVINO IR
#############################################################

In the following example:

* Original ONNX model takes one ``float32`` input with the ``{1, 3, 224, 224}`` shape, the ``RGB`` channel order, and mean/scale values applied.
* Application provides ``BGR`` image buffer with a non-fixed size and input images as batches of two.

Below is the model conversion code that can be applied in the model preparation script for this case:

* Includes / Imports


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: Python
         :fragment: ov:preprocess:save_headers

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:save_headers


* Preprocessing & Saving to the OpenVINO IR code.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: Python
         :fragment: ov:preprocess:save_model

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:save_model


Application Code - Load Model to Target Device
##############################################

Next, the application code can load a saved file and stop preprocessing. In this case, enable
:doc:`model caching <../../optimizing-latency/model-caching-overview>` to minimize load
time when the cached model is available.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.py
         :language: Python
         :fragment: ov:preprocess:save_load

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:save_load


Additional Resources
####################

* :doc:`Preprocessing Details <../preprocessing-api-details>`
* :doc:`Layout API overview <../layout-api-overview>`
* :doc:`Model Caching Overview <../../optimizing-latency/model-caching-overview>`
* :doc:`Model Preparation <../../../../model-preparation>`
* The `ov::preprocess::PrePostProcessor <https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1preprocess_1_1_pre_post_processor.html>`__ C++ class documentation
* The `ov::pass::Serialize <https://docs.openvino.ai/2025/api/c_cpp_api/classov_1_1pass_1_1_serialize.html>`__ - pass to serialize model to XML/BIN
* The ``ov::set_batch`` - update batch dimension for a given model

