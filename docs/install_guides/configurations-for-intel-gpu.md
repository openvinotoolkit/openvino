# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:



To use the OpenVINO™ GPU plugin and offload inference to Intel® Processor Graphics (GPU), Intel® Graphics Driver must be properly configured on your system.

If Intel® Graphics Driver is already installed and you would like to keep it, you can skip the installation steps below.

Linux
##########

To install the latest available **Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver** for your operating system, 
see its `installation guide on GitHub <https://github.com/intel/compute-runtime/releases/latest>`__.

.. note::
   If you use RedHat 8 OS please install OpenCL library as prerequisite via the following command line: 
   ``sh rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm``

   For instructions specific to discrete graphics platforms, refer to `the dgpu guide <https://dgpu-docs.intel.com/installation-guides/index.html>`__ 
   (Intel® Arc™ A-Series Graphics, Intel® Data Center GPU Flex Series, Intel® Data Center GPU MAX Series, Intel® processor graphics Gen12, and Intel® Iris Xe MAX codename DG1).


You may consider installing one of the earlier versions of the driver, based on your particular setup needs. For instructions and recommendations on the installation of a 
specific GPU driver release, as well as the list of supported hardware platforms, refer to the 
`Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver GitHub home page <https://github.com/intel/compute-runtime/>`__.

For instructions specific to discrete graphics platforms, refer to `the dgpu guide <https://dgpu-docs.intel.com/installation-guides/index.html>`__, 
including installation guides for Intel® Arc™ A-Series Graphics, Intel® Data Center GPU Flex Series, Intel® Data Center GPU MAX Series, Intel® processor graphics Gen12, and Intel® Iris Xe MAX codename DG1.


.. _gpu guide windows:

Windows
##########

To install the Intel Graphics Driver for Windows on your system, follow the `driver installation guide <https://www.intel.com/content/www/us/en/support/articles/000005629/graphics.html>`_.

To check if you have this driver installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.
2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:  

   .. image:: _static/images/DeviceManager.PNG
      :width: 400

3. Right-click the adapter name and select **Properties**.
4. Click the **Driver** tab to see the driver version.  

   .. image:: _static/images/DeviceDriverVersion.PNG
      :width: 400


You are done updating your device driver and ready to use your GPU.

Additional info
####################

For your reference, the following versions of Intel® Graphics Driver were used in the OpenVINO internal validation:


+------------------+-------------------------------------------------------------------------------------+
| Operation System | Driver version                                                                      |
+==================+=====================================================================================+
| Ubuntu 20.04     | `22.35.24055 <https://github.com/intel/compute-runtime/releases/tag/22.35.24055>`__ |
+------------------+-------------------------------------------------------------------------------------+
| Ubuntu 18.04     | `21.38.21026 <https://github.com/intel/compute-runtime/releases/tag/21.38.21026>`__ |
+------------------+-------------------------------------------------------------------------------------+
| CentOS 7         | `19.41.14441 <https://github.com/intel/compute-runtime/releases/tag/19.41.14441>`__ |
+------------------+-------------------------------------------------------------------------------------+
| RHEL 8           | `22.28.23726 <https://github.com/intel/compute-runtime/releases/tag/22.28.23726>`__ |
+------------------+-------------------------------------------------------------------------------------+

@endsphinxdirective


## What’s Next?

You can try out the toolkit with:

Developing in Python:
   * [Start with tensorflow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Start with ONNX and PyTorch models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html)
   * [Start with PaddlePaddle models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/103-paddle-onnx-to-openvino-classification-with-output.html)

Developing in C++:
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)

