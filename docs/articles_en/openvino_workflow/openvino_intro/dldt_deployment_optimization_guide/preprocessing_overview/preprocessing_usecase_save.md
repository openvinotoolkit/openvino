# Use Case - Integrate and Save Preprocessing Steps Into IR {#openvino_docs_OV_UG_Preprocess_Usecase_save}

@sphinxdirective

.. meta::
   :description: Once a model is read, the preprocessing/ postprocessing steps 
                 can be added and then the resulting model can be saved to 
                 OpenVINO Intermediate Representation.


Previous sections covered the topic of the :doc:`preprocessing steps <openvino_docs_OV_UG_Preprocessing_Details>` 
and the overview of :doc:`Layout <openvino_docs_OV_UG_Layout_Overview>` API.

For many applications, it is also important to minimize read/load time of a model. 
Therefore, performing integration of preprocessing steps every time on application 
startup, after ``ov::runtime::Core::read_model``, may seem inconvenient. In such cases, 
once pre and postprocessing steps have been added, it can be useful to store new execution 
model to OpenVINO Intermediate Representation (OpenVINO IR, `.xml` format).

Most available preprocessing steps can also be performed via command-line options, 
using Model Optimizer. For details on such command-line options, refer to the 
:doc:`Optimizing Preprocessing Computation <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>`.

Code example - Saving Model with Preprocessing to OpenVINO IR
#############################################################

When some preprocessing steps cannot be integrated into the execution graph using 
Model Optimizer command-line options (for example, ``YUV``->``RGB`` color space conversion, 
``Resize``, etc.), it is possible to write a simple code which:

* Reads the original model (OpenVINO IR, TensorFlow, TensorFlow Lite, ONNX, PaddlePaddle).
* Adds the preprocessing/postprocessing steps.
* Saves resulting model as IR (``.xml`` and ``.bin``).

Consider the example, where an original ONNX model takes one ``float32`` input with the 
``{1, 3, 224, 224}`` shape, the ``RGB`` channel order, and mean/scale values applied. 
In contrast, the application provides ``BGR`` image buffer with a non-fixed size and 
input images as batches of two. Below is the model conversion code that can be applied 
in the model preparation script for such a case.

* Includes / Imports


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: Python
         :fragment: ov:preprocess:save_headers

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:save_headers


* Preprocessing & Saving to the OpenVINO IR code.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: Python
         :fragment: ov:preprocess:save

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:save


Application Code - Load Model to Target Device
##############################################

After this, the application code can load a saved file and stop preprocessing. In this case, enable 
:doc:`model caching <openvino_docs_OV_UG_Model_caching_overview>` to minimize load 
time when the cached model is available.


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/ov_preprocessing.py
         :language: Python
         :fragment: ov:preprocess:save_load

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_preprocessing.cpp
         :language: cpp
         :fragment: ov:preprocess:save_load


Additional Resources
####################

* :doc:`Preprocessing Details <openvino_docs_OV_UG_Preprocessing_Details>`
* :doc:`Layout API overview <openvino_docs_OV_UG_Layout_Overview>`
* :doc:`Model Optimizer - Optimize Preprocessing Computation <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>`
* :doc:`Model Caching Overview <openvino_docs_OV_UG_Model_caching_overview>`
* The `ov::preprocess::PrePostProcessor <https://docs.openvino.ai/2023.1/classov_1_1preprocess_1_1PrePostProcessor.html#doxid-classov-1-1preprocess-1-1-pre-post-processor>`__ C++ class documentation
* The `ov::pass::Serialize <https://docs.openvino.ai/2023.1/classov_1_1pass_1_1Serialize.html#doxid-classov-1-1pass-1-1-serialize.html>`__ - pass to serialize model to XML/BIN
* The `ov::set_batch <https://docs.openvino.ai/2023.1/namespaceov.html#doxid-namespaceov-1a3314e2ff91fcc9ffec05b1a77c37862b.html>`__ - update batch dimension for a given model

@endsphinxdirective
