# Configurations for Intel® Processor Graphics (GPU) with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_configurations_for_intel_gpu}


@sphinxdirective

.. _gpu guide:

@endsphinxdirective


This page introduces additional configurations for Intel® Processor Graphics (GPU) with Intel® Distribution of OpenVINO™ toolkit on Linux and Windows.

## Linux

Once you have installed OpenVINO, follow these steps to configure it to work on GPU:

1. Download the `install_NEO_OCL_driver.sh` install script from the OpenVINO GitHub repository:
   ```sh
   curl https://raw.githubusercontent.com/openvinotoolkit/openvino/master/scripts/install_dependencies/install_NEO_OCL_driver.sh --output install_NEO_OCL_driver.sh
   ```

2. Install the **Intel® Graphics Compute Runtime for OpenCL™** driver components required to use the GPU plugin and write custom layers for Intel® Integrated Graphics. These drivers are not included in the OpenVINO installation package, so the install script downloads and installs them separately. To install, run the script:
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

You are done updating your device driver and are ready to use your GPU. Proceed to the <a href="openvino_docs_install_guides_installing_openvino_windows.html#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.

