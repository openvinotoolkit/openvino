Hello Remote Tensor Sample
==========================


.. meta::
   :description: Learn how to run image classification on an Intel GPU
                 using Level Zero remote tensors (C++).


This sample demonstrates image classification on an Intel GPU with input and
output tensors allocated in Level Zero device memory. Requirements:

- An Intel GPU with a driver that provides the Level Zero runtime.
- OpenVINO built with the experimental Level Zero GPU plugin
  (``GPU_RT_TYPE=ZE`` or ``GPU_RT_TYPE=COMBINED``).
- Level Zero and OpenCL development files. CMake skips this sample if either
  is missing.
- A model with one input and one output, in any format supported by
  ``ov::Core::read_model``.
- A directly selected GPU device, for example ``GPU`` or ``GPU.0``.
  CPU, AUTO, MULTI, and HETERO are not supported.
- To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>`
  section in "Get Started with Samples" guide.

How It Works
####################

The sample is a Level Zero variant of :doc:`Hello Classification <hello-classification>`.
It compiles the model in the GPU plugin's default remote context, creates
``USM_DEVICE_BUFFER`` remote tensors for the input and output, and moves data
between host and device memory with a Level Zero command list. On drivers with
Level Zero - OpenCL interoperability (LEO), the default context is an OpenCL
context, so the underlying Level Zero context is queried with ``clGetContextInfo``.

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. scrollbox::

         .. doxygensnippet:: samples/cpp/hello_remote_tensor/main.cpp
            :language: cpp


Running
####################

.. code-block:: console

   hello_remote_tensor <path_to_model> <path_to_image> <device_name>

For example:

.. code-block:: console

   hello_remote_tensor ./models/bvlcalexnet-12.onnx ./images/dog.png GPU

.. note::

   - The sample expects input with BGR channel order and the NCHW layout for
     the model input, as described in :doc:`Hello Classification <hello-classification>`.
   - Paths may contain Unicode characters.

The application logs model information and prints the top 10 classification
results.

Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <../../../openvino-workflow/running-inference>`
- :doc:`Get Started with Samples <get-started-demos>`
- :doc:`Using OpenVINO Samples <../openvino-samples>`
- `Hello Remote Tensor C++ Sample on Github <https://github.com/openvinotoolkit/openvino/blob/master/samples/cpp/hello_remote_tensor/README.md>`__
