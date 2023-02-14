# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:


To use the OpenVINO™ GPU plugin and offload inference to Intel® Processor Graphics (GPU), Intel® Graphics Driver must be properly configured on your system.

If Intel® Graphics Driver is already installed and you would like to keep it, you can skip the installation steps below.

@endsphinxdirective

## Linux

@sphinxdirective

To install the latest available **Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver** for your operating system, see its `installation guide on GitHub <https://github.com/intel/compute-runtime/releases/latest>`_.

.. note::

   If you are using RedHat 8, you may install the OpenCL library as a prerequisite by using the following command:

   .. code-block:: sh

      rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm


You can also install an earlier version of the driver, based on your particular needs for setup.

For instructions and recommendations on the installation of a specific GPU driver release, as well as the list of supported hardware platforms, refer to the `Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver GitHub home page <https://github.com/intel/compute-runtime/>`_.

For specific instructions for discrete graphics platforms, refer to `the dGPU guides <https://dgpu-docs.intel.com/installation-guides/index.html>`_, including installation guides for Intel® Arc™ A-Series Graphics, Intel® Data Center GPU Flex Series, Intel® Data Center GPU MAX Series, Intel® processor graphics Gen12, and Intel® Iris Xe MAX codename DG1.

.. _gpu guide windows:

@endsphinxdirective

## Windows

@sphinxdirective

To install Intel® Graphics Driver for Windows on your hardware, follow the instructions on `this article on Intel® Support site <https://www.intel.com/content/www/us/en/support/articles/000005629/graphics.html>`_. 

To check whether you have this driver installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.
2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:

   .. image:: _static/images/DeviceManager.png
      :width: 400

3. Right-click the adapter name and select **Properties**.
4. Click the **Driver** tab to see the driver version.  

   .. image:: _static/images/DeviceDriverVersion.png
      :width: 400


You are done updating your device driver and are ready to use your GPU.

@endsphinxdirective

## Additional Information

@sphinxdirective

For your reference, the following versions of Intel® Graphics Driver were used in the OpenVINO internal validation:

+------------------+-------------------------------------------------------------------------------------+
| Operating System || Driver version                                                                     |
+==================+=====================================================================================+
| Ubuntu 20.04     || `22.35.24055 <https://github.com/intel/compute-runtime/releases/tag/22.35.24055>`_ |
+------------------+------------------------------------------------------+------------------------------+
| Ubuntu 18.04     || `21.38.21026 <https://github.com/intel/compute-runtime/releases/tag/21.38.21026>`_ |
+------------------+------------------------------------------------------+------------------------------+
| CentOS 7         || `19.41.14441 <https://github.com/intel/compute-runtime/releases/tag/19.41.14441>`_ |
+------------------+------------------------------------------------------+------------------------------+
| RHEL 8           || `22.28.23726 <https://github.com/intel/compute-runtime/releases/tag/22.28.23726>`_ |
+------------------+-------------------------------------------------------------------------------------+

@endsphinxdirective

## What’s Next?

@sphinxdirective

.. tab:: Get started with Python

   Try the `Python Quick Start Example <https://docs.openvino.ai/nightly/notebooks/201-vision-monodepth-with-output.html>`_ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.
   
   .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
      :width: 400

   Visit the :ref:`Tutorials <notebook tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:
   
   * `OpenVINO Python API Tutorial <https://docs.openvino.ai/nightly/notebooks/002-openvino-api-with-output.html>`_
   * `Basic image classification program with Hello Image Classification <https://docs.openvino.ai/nightly/notebooks/001-hello-world-with-output.html>`_
   * `Convert a PyTorch model and use it for image background removal <https://docs.openvino.ai/nightly/notebooks/205-vision-background-removal-with-output.html>`_

.. tab:: Get started with C++

   Try the `C++ Quick Start Example <openvino_docs_get_started_get_started_demos.html>`_ for step-by-step instructions on building and running a basic image classification C++ application.
   
   .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
      :width: 400

   Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:
   
   * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_inference_engine_samples_hello_reshape_ssd_README.html>`_
   * `Automatic speech recognition C++ sample <openvino_inference_engine_samples_speech_sample_README.html>`_

@endsphinxdirective