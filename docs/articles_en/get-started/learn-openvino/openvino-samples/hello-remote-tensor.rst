Hello Remote Tensor Sample
==========================


.. meta::
   :description: Learn how to run image classification on an Intel GPU
                 using Level Zero remote tensors (C++).


This sample demonstrates how to integrate OpenVINO into an existing Level Zero
application. It runs image classification on an Intel GPU with input and output
tensors allocated in Level Zero device memory. OpenVINO does not provide a
dedicated Level Zero remote context, so the sample uses Level Zero - OpenCL
interoperability (LEO) of the GPU driver to convert the GPU plugin's OpenCL
context to its Level Zero context. Requirements:

- An Intel GPU with a driver that supports LEO.
- OpenVINO with the GPU plugin, built with any runtime (``GPU_RT_TYPE=OCL``,
  ``ZE`` or ``COMBINED``).
- Level Zero and OpenCL development files. CMake skips this sample if either
  is missing. At runtime, the Level Zero loader library must be installed in
  the system (for example, the ``libze1`` package on Ubuntu).
- A model with one input and one output, in any format supported by
  ``ov::Core::read_model``.
- A directly selected GPU device, for example ``GPU`` or ``GPU.0``.
  CPU, AUTO, MULTI, and HETERO are not supported.
- To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>`
  section in "Get Started with Samples" guide.

How It Works
####################

The sample is a Level Zero variant of :doc:`Hello Classification <hello-classification>`.
It takes the GPU plugin's default remote context, which is an OpenCL context,
and converts it to the underlying Level Zero context with the LEO query
``clGetContextInfo(..., CL_L0_CONTEXT_HANDLE, ...)``. Then it compiles the model
in that context, creates ``USM_DEVICE_BUFFER`` remote tensors for the input and
output, and moves data between host and device memory with a Level Zero command
list. USM device pointers of the remote tensors are shared by OpenCL and Level
Zero, so they need no conversion. Only when the Level Zero GPU plugin runs on a
driver without LEO, the default context is already a Level Zero context
(``ov::intel_gpu::ContextType::ZE``) and is used directly.

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
