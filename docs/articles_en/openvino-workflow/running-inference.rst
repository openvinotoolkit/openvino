Running and Integrating Inference Pipeline
===============================================================================================

.. meta::
   :description: Learn how to implement a typical inference pipeline of OpenVINO™
                 Runtime in an application.⠀


.. toctree::
   :maxdepth: 1
   :hidden:

   running-inference/model-representation
   running-inference/model-input-output
   running-inference/inference-request
   running-inference/stateful-models
   running-inference/python-api-advanced-inference
   running-inference/python-api-exclusives
   running-inference/inference-devices-and-modes
   Optimize Inference <running-inference/optimize-inference>


OpenVINO Runtime is a set of C++ libraries with C and Python bindings, providing a common
API to run inference on various devices. Each device (integrated with OpenVINO’s plugin
architecture) offers the common, as well as hardware-specific APIs for more configuration
options. Note that OpenVINO Runtime may also be integrated with other frameworks and work
as their backend, for example, using torch.compile.
The scheme below illustrates the typical workflow for deploying a trained deep learning
model in an application:

.. image:: ../assets/images/IMPLEMENT_PIPELINE_with_API_C.svg

This guide will show you how to implement a typical OpenVINO™ Runtime inference pipeline
in your application. Before proceeding, check how
:doc:`model conversion <model-preparation/convert-model-to-ir>`
works in OpenVINO and how it may affect your applications’ performance. Make sure you have
installed OpenVINO Runtime and set environment variables (otherwise, the ``find_package``
calls will not find OpenVINO_DIR):

.. tab-set::

   .. tab-item:: Linux
      :sync: lin

      .. code-block:: console

         <INSTALL_DIR>/setupvars.sh


   .. tab-item:: Windows
      :sync: win


      PowerShell:

      .. code-block:: console

         <INSTALL_DIR>/setupvars.sh

      Command Prompt

      .. code-block:: console

         cd  <INSTALL_DIR>
         setupvars.bat




Step 1. Create OpenVINO Runtime Core
###############################################################################################

Initiate working with OpenVINO in your application by including the OpenVINO™ Runtime
components:

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
###############################################################################################

Compile the model with ``ov::Core::compile_model()``, defining the device or mode to use
for inference. The following example uses the
:doc:`AUTO mode <running-inference/inference-devices-and-modes/auto-device-selection>`,
which selects the device for you. To learn more about supported devices and inference modes,
see the :doc:`Inference Devices and Modes <running-inference/inference-devices-and-modes>`
section.

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


The ``ov::CompiledModel`` class represents a compiled model and enables you to get
information inputs or output ports by a tensor name or index. This approach is aligned with
most frameworks. The ``ov::Model`` object represents any models inside the OpenVINO™ Runtime.
For more details, refer to
:doc:`OpenVINO™ Model representation <running-inference/model-representation>`.

Step 3. Create an Inference Request
###############################################################################################

Use the ``ov::InferRequest`` class methods to create an infer request. For more details,
see the article on
:doc:`InferRequest <running-inference/inference-request>`.

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
###############################################################################################

Create ``ov::Tensor``, you can use external memory for that , and use the
``ov::InferRequest::set_input_tensor`` method to send this tensor to the device.
For more info on textual data as input, see the
:doc:`String Tensors <running-inference/model-input-output/string-tensors>` article.

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
###############################################################################################

Use either ``ov::InferRequest::start_async`` or ``ov::infer_request.infer()`` to start model
inference. To learn how they work, see the
:doc:`OpenVINO Inference Request <running-inference/inference-request>`
article. The following example uses the asynchronous option and calls
``ov::InferRequest::wait`` to wait for the inference results.

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


Step 6. Process the Inference Results
#################################################################################################

Get output tensors and process the inference results.
For more info on textual data as input, see the
:doc:`String Tensors <running-inference/model-input-output/string-tensors>` article.

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


Step 7. [only for C] Release the allocated objects
###############################################################################################

To avoid memory leak, applications developed with the C API need to release the allocated
objects in the following order.

.. tab-set::

    .. tab-item:: C
       :sync: c

       .. doxygensnippet:: docs/snippets/src/main.c
          :language: cpp
          :fragment: [part8]


Build Your Application
###############################################################################################

If you have integrated OpenVINO with your application, you will need to adjust your
application build process as well. Of course, there are multiple ways this stage may be
done, so you will need to choose the one best for your project. To learn about the basics of
OpenVINO build process, refer to the
`documentation on GitHub <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__.

The following example uses a C++ & C application together with CMake,
for project configuration.

1. Create Structure for project:

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


2. Configure the CMake build

   For details on additional CMake build options, refer to the
   `CMake page <https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)>`__.

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

3. Build Project

   Use CMake to build the project on your system:

   .. code-block:: sh

      cd build/
      cmake ../project
      cmake --build .


Additional Resources
####################

* To see working implementation of the steps, check out the
  :doc:`Learn OpenVINO <../get-started/learn-openvino>` section, including
  `OpenVINO™ Runtime API Tutorial <./../../notebooks/openvino-api-with-output.html>`__.
* Models in the OpenVINO IR format on `Hugging Face <https://huggingface.co/models>`__.
* :ref:`Using Encrypted Models with OpenVINO <encrypted-models>`
