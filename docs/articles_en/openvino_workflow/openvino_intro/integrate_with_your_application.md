# Integrate OpenVINO™ with Your Application {#openvino_docs_OV_UG_Integrate_OV_with_your_application}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Model_Representation
   openvino_docs_OV_UG_Infer_request
   openvino_docs_OV_UG_Python_API_inference
   openvino_docs_OV_UG_Python_API_exclusives
   openvino_docs_MO_DG_TensorFlow_Frontend


.. meta::
   :description: Learn how to implement a typical inference pipeline of OpenVINO™ 
                 Runtime in an application.


Following these steps, you can implement a typical OpenVINO™ Runtime inference 
pipeline in your application. Before proceeding, make sure you have 
:doc:`installed OpenVINO Runtime <openvino_docs_install_guides_overview>` and set environment variables (run ``<INSTALL_DIR>/setupvars.sh`` for Linux or ``setupvars.bat`` for Windows, otherwise, the ``OpenVINO_DIR`` variable won't be configured properly to pass ``find_package`` calls).


.. image:: _static/images/IMPLEMENT_PIPELINE_with_API_C.svg


Step 1. Create OpenVINO Runtime Core
####################################

Include next files to work with OpenVINO™ Runtime:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [import]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [include]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [include]


Use the following code to create OpenVINO™ Core to manage available devices and read model objects:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [part1]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part1]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part1]


Step 2. Compile the Model
#########################

``ov::CompiledModel`` class represents a device specific compiled model. ``ov::CompiledModel`` allows you to get information inputs or output ports by a tensor name or index. This approach is aligned with the majority of frameworks.

Compile the model for a specific device using ``ov::Core::compile_model()``:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. tab-set::

          .. tab-item:: IR
             :sync: ir

             .. doxygensnippet:: docs/snippets/src/main.py
                :language: python
                :fragment: [part2_1]

          .. tab-item:: ONNX
             :sync: onnx

             .. doxygensnippet:: docs/snippets/src/main.py
                :language: python
                :fragment: [part2_2]

          .. tab-item:: PaddlePaddle
             :sync: paddlepaddle

             .. doxygensnippet:: docs/snippets/src/main.py
                :language: python
                :fragment: [part2_3]

          .. tab-item:: TensorFlow
             :sync: tensorflow

             .. doxygensnippet:: docs/snippets/src/main.py
                :language: python
                :fragment: [part2_4]

          .. tab-item:: TensorFlow Lite
             :sync: tensorflow_lite

             .. doxygensnippet:: docs/snippets/src/main.py
                :language: python
                :fragment: [part2_5]

          .. tab-item:: ov::Model
             :sync: openvinomodel

             .. doxygensnippet:: docs/snippets/src/main.py
                :language: python
                :fragment: [part2_6]

    .. tab-item:: C++
       :sync: cpp

       .. tab-set::

          .. tab-item:: IR
             :sync: ir

             .. doxygensnippet:: docs/snippets/src/main.cpp
                :language: cpp
                :fragment: [part2_1]

          .. tab-item:: ONNX
             :sync: onnx

             .. doxygensnippet:: docs/snippets/src/main.cpp
                :language: cpp
                :fragment: [part2_2]

          .. tab-item:: PaddlePaddle
             :sync: paddlepaddle

             .. doxygensnippet:: docs/snippets/src/main.cpp
                :language: cpp
                :fragment: [part2_3]

          .. tab-item:: TensorFlow
             :sync: tensorflow

             .. doxygensnippet:: docs/snippets/src/main.cpp
                :language: cpp
                :fragment: [part2_4]

          .. tab-item:: TensorFlow Lite
             :sync: tensorflow_lite

             .. doxygensnippet:: docs/snippets/src/main.cpp
                :language: cpp
                :fragment: [part2_5]

          .. tab-item:: ov::Model
             :sync: openvinomodel

             .. doxygensnippet:: docs/snippets/src/main.cpp
                :language: cpp
                :fragment: [part2_6]

    .. tab-item:: C
       :sync: c

       .. tab-set::

          .. tab-item:: IR
             :sync: ir

             .. doxygensnippet:: docs/snippets/src/main.c
                :language: cpp
                :fragment: [part2_1]

          .. tab-item:: ONNX
             :sync: onnx

             .. doxygensnippet:: docs/snippets/src/main.c
                :language: cpp
                :fragment: [part2_2]

          .. tab-item:: PaddlePaddle
             :sync: paddlepaddle

             .. doxygensnippet:: docs/snippets/src/main.c
                :language: cpp
                :fragment: [part2_3]

          .. tab-item:: TensorFlow
             :sync: tensorflow

             .. doxygensnippet:: docs/snippets/src/main.c
                :language: cpp
                :fragment: [part2_4]

          .. tab-item:: TensorFlow Lite
             :sync: tensorflow_lite

             .. doxygensnippet:: docs/snippets/src/main.c
                :language: cpp
                :fragment: [part2_5]

          .. tab-item:: ov::Model
             :sync: openvinomodel

             .. doxygensnippet:: docs/snippets/src/main.c
                :language: cpp
                :fragment: [part2_6]


The ``ov::Model`` object represents any models inside the OpenVINO™ Runtime.
For more details please read article about :doc:`OpenVINO™ Model representation <openvino_docs_OV_UG_Model_Representation>`.

The code above creates a compiled model associated with a single hardware device from the model object.
It is possible to create as many compiled models as needed and use them simultaneously (up to the limitation of the hardware resources).
To learn how to change the device configuration, read the :doc:`Query device properties <openvino_docs_OV_UG_query_api>` article.

Step 3. Create an Inference Request
###################################

``ov::InferRequest`` class provides methods for model inference in OpenVINO™ Runtime. 
Create an infer request using the following code (see 
:doc:`InferRequest detailed documentation <openvino_docs_OV_UG_Infer_request>` for more details):

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [part3]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part3]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part3]


Step 4. Set Inputs
####################

You can use external memory to create ``ov::Tensor`` and use the ``ov::InferRequest::set_input_tensor`` method to put this tensor on the device:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [part4]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part4]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part4]


Step 5. Start Inference
#######################

OpenVINO™ Runtime supports inference in either synchronous or asynchronous mode. Using the Async API can improve application's overall frame-rate: instead of waiting for inference to complete, the app can keep working on the host while the accelerator is busy. You can use ``ov::InferRequest::start_async`` to start model inference in the asynchronous mode and call ``ov::InferRequest::wait`` to wait for the inference results:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [part5]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part5]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part5]


This section demonstrates a simple pipeline. To get more information about other ways to perform inference, read the dedicated 
:doc:`"Run inference" section <openvino_docs_OV_UG_Infer_request>`.

Step 6. Process the Inference Results
#####################################

Go over the output tensors and process the inference results.

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [part6]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part6]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part6]


Step 7. Release the allocated objects (only for C)
##################################################

To avoid memory leak, applications developed with C API need to release the allocated objects in order.

.. tab-set::

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part8]


Step 8. Link and Build Your Application with OpenVINO™ Runtime (example)
########################################################################

This step may differ for different projects. In this example, a C++ & C application is used, together with CMake for project configuration.

Create Structure for project:
+++++++++++++++++++++++++++++

.. tab-set::

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part7]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part7]


Create Cmake Script
++++++++++++++++++++

For details on additional CMake build options, refer to the `CMake page <https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)>`__.

.. tab-set::

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/CMakeLists.txt
          :language: cpp
          :fragment: [cmake:integration_example_cpp]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/CMakeLists.txt
          :language: cpp
          :fragment: [cmake:integration_example_c]


Build Project
++++++++++++++++++++

To build your project using CMake with the default build tools currently available on your machine, execute the following commands:

.. code-block:: sh

   cd build/
   cmake ../project
   cmake --build .


Additional Resources
####################

* See the :doc:`OpenVINO Samples <openvino_docs_OV_UG_Samples_Overview>` page or the `Open Model Zoo Demos <https://docs.openvino.ai/2023.1/omz_demos.html>`__ page for specific examples of how OpenVINO pipelines are implemented for applications like image classification, text prediction, and many others.
* :doc:`OpenVINO™ Runtime Preprocessing <openvino_docs_OV_UG_Preprocessing_Overview>`
* :doc:`Using Encrypted Models with OpenVINO <openvino_docs_OV_UG_protecting_model_guide>`
* `Open Model Zoo Demos <https://docs.openvino.ai/2023.1/omz_demos.html>`__

@endsphinxdirective
