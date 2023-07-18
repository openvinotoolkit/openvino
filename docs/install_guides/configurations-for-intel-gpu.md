# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:

To use the OpenVINO™ GPU plug-in and transfer the inference to the graphics of the Intel® processor (GPU), the Intel® graphics driver must be properly configured on the system.

Linux
##########

To use a GPU device for OpenVINO inference, you must install OpenCL runtime packages.

If you are using a discrete GPU (for example Arc 770), you must also be using a supported Linux kernel as per `documentation. <https://dgpu-docs.intel.com/driver/kernel-driver-types.html>`__ 

- For Arc GPU, kernel 6.2 or higher is recommended. 
- For Max and Flex GPU, or Arc with kernel version lower than 6.2, you must also install the ``intel-i915-dkms`` and ``xpu-smi`` kernel modules as described in the installation documentation for `Max/Flex <https://dgpu-docs.intel.com/driver/installation.html>`__ or `Arc. <https://dgpu-docs.intel.com/driver/client/overview.html>`__

Below are the instructions on how to install the OpenCL packages on supported Linux distributions. These instructions install the `Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver <https://github.com/intel/compute-runtime/releases/tag/23.22.26516.18>`__ and its dependencies: 

- `Intel Graphics Memory Management Library <https://github.com/intel/gmmlib>`__
- `Intel® Graphics Compiler for OpenCL™ <https://github.com/intel/intel-graphics-compiler>`__
- `OpenCL ICD loader package <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`__

.. tab-set::

   .. tab-item:: Ubuntu 22.04 LTS
      :sync: ubuntu-22

      Download and install the `deb` packages published `here <https://github.com/intel/compute-runtime/releases/latest>`__ and install the apt package `ocl-icd-libopencl1` with the OpenCl ICD loader.
      
      Alternatively, you can add the apt repository by following the `installation guide <https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps>`__. Then install the `ocl-icd-libopencl1`, `intel-opencl-icd`, `intel-level-zero-gpu` and `level-zero` apt packages:
      
      .. code-block:: sh
      
         apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero

   .. tab-item:: Ubuntu 20.04 LTS
      :sync: ubuntu-20

      Ubuntu 20.04 LTS is not updated with the latest driver versions. You can install the updated versions up to the version 22.43 from apt:
      
      .. code-block:: sh
         
         apt-get update && apt-get install -y --no-install-recommends curl gpg gpg-agent && \
         curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
         echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
         apt-get update
         apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero
      
      Alternatively, download older `deb` version from `here <https://github.com/intel/compute-runtime/releases>`__. Note that older driver version might not include some of the bug fixes and might be not supported on some latest platforms. Check the supported hardware for the versions you are installing.

   .. tab-item:: RedHat UBI 8
      :sync: redhat-8

      Follow the `guide <https://dgpu-docs.intel.com/driver/installation.html#rhel-install-steps>`__ to add Yum repository.
      
      Install following packages: 
      
      .. code-block:: sh
      
         yum install intel-opencl level-zero intel-level-zero-gpu intel-igc-core intel-igc-cm intel-gmmlib intel-ocloc
      
      Install the OpenCL ICD Loader via:
      
      .. code-block:: sh
      
         rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm

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

+------------------+-------------------------------------------------------------------------------------------+
| Operation System | Driver version                                                                            |
+==================+===========================================================================================+
| Ubuntu 22.04     | `22.43.24595.30 <https://github.com/intel/compute-runtime/releases/tag/22.43.24595.30>`__ |
+------------------+-------------------------------------------------------------------------------------------+
| Ubuntu 20.04     | `22.35.24055 <https://github.com/intel/compute-runtime/releases/tag/22.35.24055>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| Ubuntu 18.04     | `21.38.21026 <https://github.com/intel/compute-runtime/releases/tag/21.38.21026>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| CentOS 7         | `19.41.14441 <https://github.com/intel/compute-runtime/releases/tag/19.41.14441>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| RHEL 8           | `22.28.23726 <https://github.com/intel/compute-runtime/releases/tag/22.28.23726>`__       |
+------------------+-------------------------------------------------------------------------------------------+

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

