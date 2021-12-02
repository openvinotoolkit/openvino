# GPU Setup Guide for Use with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_gpu_setup_guide}

@sphinxdirective

.. _gpu guide:

@endsphinxdirective

Once you have your OpenVINO installed, follow the steps to be able to work on GPU:

1. Go to the install_dependencies directory:
   ```sh
   cd <INSTALL_DIR>/intel/openvino_2022/install_dependencies/
   ```

2. Install the **Intel® Graphics Compute Runtime for OpenCL™** driver components required to use the GPU plugin and write custom layers for Intel® Integrated Graphics. The drivers are not included in the package. To install, run this script:
   ```sh
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
Proceed to the <a href="#get-started">Start Using the Toolkit</a> section to learn the basic OpenVINO™ toolkit workflow and run code samples and demo applications.
