Integrate OpenVINO™ with Your Application
===========================================


.. toctree::
   :maxdepth: 1
   :hidden:

   integrate-openvino-with-your-application/model-representation
   integrate-openvino-with-your-application/inference-request
   integrate-openvino-with-your-application/python-api-advanced-inference
   integrate-openvino-with-your-application/python-api-exclusives


.. meta::
   :description: Learn how to implement a typical inference pipeline of OpenVINO™
                 Runtime in an application.


Following these steps, you can implement a typical OpenVINO™ Runtime inference pipeline in your
application. Before proceeding, make sure you have :doc:`installed OpenVINO Runtime <../../get-started/install-openvino>`
and set environment variables (run ``<INSTALL_DIR>/setupvars.sh`` for Linux, ``setupvars.ps1``
for Windows PowerShell, or ``setupvars.bat`` for Windows CMD). Otherwise, the ``OpenVINO_DIR``
variable won't be configured properly to pass ``find_package`` calls.


.. image:: ../../assets/images/IMPLEMENT_PIPELINE_with_API_C.svg


Step 1. Create OpenVINO Runtime Core
####################################

Include the necessary files to work with OpenVINO™ Runtime and create OpenVINO™ Core to manage
available devices and read model objects:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [import]

       .. doxygensnippet:: docs/snippets/src/main.py
          :language: python
          :fragment: [part1]

    .. tab-item:: C++
       :sync: cpp

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [include]

       .. doxygensnippet:: docs/snippets/src/main.cpp
          :language: cpp
          :fragment: [part1]

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [include]

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part1]



Step 2. Compile the Model
#########################

``ov::CompiledModel`` class represents a device specific compiled model. ``ov::CompiledModel`` allows you to get information inputs or output ports by a tensor name or index. This approach is aligned with the majority of frameworks.
:doc:`AUTO mode <./inference-devices-and-modes/auto-device-selection>` automatically selects the most suitable hardware for running inference.

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
For more details please read article about :doc:`OpenVINO™ Model representation <integrate-openvino-with-your-application/model-representation>`.

The code above creates a compiled model associated with a single hardware device from the model object.
It is possible to create as many compiled models as needed and use them simultaneously (up to the limitation of the hardware).
To learn more about supported devices and inference modes, read the :doc:`Inference Devices and Modes <./inference-devices-and-modes>` article.


Step 3. Create an Inference Request
###################################

``ov::InferRequest`` class provides methods for model inference in OpenVINO™ Runtime.
Create an infer request using the following code (see
:doc:`InferRequest documentation <integrate-openvino-with-your-application/inference-request>` for more details):

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

See :doc:`additional materials <string-tensors>` to learn how to handle textual data as a model input.

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
:doc:`"Run inference" section <integrate-openvino-with-your-application/inference-request>`.

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

See :doc:`additional materials <string-tensors>` to learn how to handle textual data as a model output.

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
          :force:

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part7]
          :force:


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
          
    .. tab-item:: C++ (PyPI)
      :sync: cpp 
      
      .. doxygensnippet:: docs/snippets/CMakeLists.txt
         :language: cpp
         :fragment: [cmake:integration_example_cpp_py]

Build Project
++++++++++++++++++++

To build your project using CMake with the default build tools currently available on your machine, execute the following commands:

.. code-block:: sh

   cd build/
   cmake ../project
   cmake --build .


Additional Resources
####################

* `OpenVINO™ Runtime API Tutorial <./../../notebooks/openvino-api-with-output.html>`__
* See the :doc:`OpenVINO Samples <../../learn-openvino/openvino-samples>` page for specific examples of how OpenVINO pipelines are implemented for applications like image classification, text prediction, and many others.
* Models in the OpenVINO IR format on `Hugging Face <https://huggingface.co/models>`__.
* :doc:`OpenVINO™ Runtime Preprocessing <optimize-inference/optimize-preprocessing>`
* :doc:`String Tensors <string-tensors>`
* :ref:`Using Encrypted Models with OpenVINO <encrypted-models>`

