# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:

@endsphinxdirective



## Linux

Once you have installed OpenVINO, to work with GPU, you need to install the **Intel® Graphics Compute Runtime for OpenCL™** driver components required to use the GPU plugin and write custom layers for Intel® Integrated Graphics. Those drivers are not included in the OpenVINO installation package, so OpenVINO provides a script to download and install them separately.

Do the following steps:

1. If you installed OpenVINO Runtime from the archive file, you can find the `install_NEO_OCL_driver.sh` script in the following directory:
   ```sh
   cd <INSTALL_DIR>/install_dependencies/
   ```
   
   If you installed OpenVINO Runtime via PyPI, you can download the `install_NEO_OCL_driver.sh` script from the OpenVINO GitHub repository:
   ```sh
   curl -L https://raw.githubusercontent.com/openvinotoolkit/openvino/releases/2022/2/scripts/install_dependencies/install_NEO_OCL_driver.sh --output install_NEO_OCL_driver.sh
   ```
   
2. Run the script:
   ```sh
   chmod +x install_NEO_OCL_driver.sh
   sudo -E ./install_NEO_OCL_driver.sh
   ```
   > **NOTE**: To use the **Intel® Iris® Xe MAX Graphics**, see the [Intel® Iris® Xe MAX Graphics with Linux*](https://dgpu-docs.intel.com/devices/iris-xe-max-graphics/index.html) page for driver installation instructions.

   The script compares the driver version on the system to the current version. If the driver version on the system is higher or equal to the current version, the script does
   not install a new driver. If the version of the driver is lower than the current version, the script uninstalls the lower version and installs the current version with your permission:
   ![](../img/NEO_check_agreement.png)

   Higher hardware versions require a higher driver version, namely 20.35 instead of 19.41. If the script fails to uninstall the driver, uninstall it manually. During the script execution, you may see the following command line output:  
   ```sh
   Add OpenCL user to video group    
   ```
   Ignore this suggestion and continue.<br>
   You can also find the most recent version of the driver, installation procedure and other information on the [Intel® software for general purpose GPU capabilities](https://dgpu-docs.intel.com/index.html) site.

3. **Optional:** Install header files to allow compilation of new code. You can find the header files at [Khronos OpenCL™ API Headers](https://github.com/KhronosGroup/OpenCL-Headers.git).

You've completed all required configuration steps to perform inference on processor graphics.
Proceed to the <a href="openvino_docs_install_guides_installing_openvino_linux.html#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

@sphinxdirective

.. _gpu guide windows:

@endsphinxdirective

## Windows

This section will help you check if you require driver installation. Install indicated version or higher.

If your applications offload computation to **Intel® Integrated Graphics**, you must have the Intel Graphics Driver for Windows installed on your hardware.
[Download and install the recommended version](https://downloadcenter.intel.com/download/30079/Intel-Graphics-Windows-10-DCH-Drivers).

To check if you have this driver installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.

2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:
   ![](../img/DeviceManager.PNG)

3. Right-click the adapter name and select **Properties**.

4. Click the **Driver** tab to see the driver version.
   ![](../img/DeviceDriverVersion.PNG)

You are done updating your device driver and are ready to use your GPU.
## What’s Next?

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

@sphinxdirective
.. tab:: Get started with Python

   Try the `Python Quick Start Example <https://docs.openvino.ai/2022.1/notebooks/201-vision-monodepth-with-output.html>`_ to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

   .. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
      :width: 400

   Visit the :ref:`Tutorials <notebook tutorials>` page for more Jupyter Notebooks to get you started with OpenVINO, such as:

   * `OpenVINO Python API Tutorial <https://docs.openvino.ai/2022.1/notebooks/002-openvino-api-with-output.html>`_
   * `Basic image classification program with Hello Image Classification <https://docs.openvino.ai/2022.1/notebooks/001-hello-world-with-output.html>`_
   * `Convert a PyTorch model and use it for image background removal <https://docs.openvino.ai/2022.1/notebooks/205-vision-background-removal-with-output.html>`_

.. tab:: Get started with C++

   Try the `C++ Quick Start Example <openvino_docs_get_started_get_started_demos.html>`_ for step-by-step instructions on building and running a basic image classification C++ application.

   .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
      :width: 400

   Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:

   * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_inference_engine_samples_hello_reshape_ssd_README.html>`_
   * `Automatic speech recognition C++ sample <openvino_inference_engine_samples_speech_sample_README.html>`_

@endsphinxdirective
