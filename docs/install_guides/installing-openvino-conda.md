# Install Intel® Distribution of OpenVINO™ toolkit from Anaconda* Cloud {#openvino_docs_install_guides_installing_openvino_conda}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit distributed through the Anaconda* Cloud.


## System Requirements

 - [Anaconda* distribution](https://www.anaconda.com/products/individual/)

**Operating Systems**

- Ubuntu* 18.04 long-term support (LTS), 64-bit
- CentOS* 7.4, 64-bit
- macOS* 10.14.x versions. 
- Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or higher) editions
- Windows Server* 2016 or higher



## Install the runtime package using the Anaconda* Package Manager

1. Set up the Anaconda* environment. 

2. Updated conda to the latest version:
   ```sh
   conda update --all
   ```
3. Install the Intel® Distribution of OpenVINO™ Toolkit:
 - Ubuntu* 18.04 
   ```sh
   conda install openvino-ie4py-ubuntu18 -c intel
   ```
 - CentOS* 7.4 
   ```sh
   conda install openvino-ie4py-centos7 -c intel
   ```
 - Windows* 10 and macOS*
   ```sh
   conda install openvino-ie4py -c intel
   ```
4. Verify the package installed:
   ```sh
   python -c "import openvino"
   ```
   
Now you can start to develop and run your application.


## Known Issues and Limitations

- You cannot use Python bindings included in  Intel® Distribution of OpenVINO™ toolkit  with  [Anaconda* distribution](https://www.anaconda.com/products/individual/)
- You cannot use Python OpenVINO™ bindings included in Anaconda* package with official  [Python distribution](https://https://www.python.org/).


## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit).
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org).
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- For information on Inference Engine Tutorials, see the [Inference Tutorials](https://github.com/intel-iot-devkit/inference-tutorials-generic).
- Intel® Distribution of OpenVINO™ toolkit Anaconda* home page: [https://anaconda.org/intel/openvino-ie4py](https://anaconda.org/intel/openvino-ie4py)

