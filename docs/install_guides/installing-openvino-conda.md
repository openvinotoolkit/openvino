# Install Intel® Distribution of OpenVINO™ toolkit from Anaconda* Cloud {#openvino_docs_install_guides_installing_openvino_conda}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit distributed through the Anaconda* Cloud.


## System Requirements

**Software**

 - [Anaconda* distribution](https://www.anaconda.com/products/individual/)

**Operating Systems**

| Supported Operating System                                   | [Python* Version (64-bit)](https://www.python.org/) |
| :------------------------------------------------------------| :---------------------------------------------------|
|   Ubuntu* 18.04 long-term support (LTS), 64-bit              | 3.6, 3.7                                            |
|   Ubuntu* 20.04 long-term support (LTS), 64-bit              | 3.6, 3.7                                            |
|   CentOS* 7, 64-bit                                          | 3.6, 3.7                                            |
|   macOS* 10.15.x versions                                    | 3.6, 3.7                                            |
|   Windows 10*, 64-bit                                        | 3.6, 3.7                                            |

## Install the runtime package using the Anaconda* Package Manager

1. Set up the Anaconda* environment: 
   ```sh
   conda create --name py37 python=3.7
   ```
   ```sh
   conda activate py37
   ```
2. Updated conda to the latest version:
   ```sh
   conda update --all
   ```
3. Install the Intel® Distribution of OpenVINO™ Toolkit:
 - Ubuntu* 18.04 
   ```sh
   conda install openvino-ie4py-ubuntu18 -c intel
   ```
 - CentOS* 7.6 
   ```sh
   conda install openvino-ie4py-centos7 -c intel
   ```
 - Windows* 10 and macOS*
   ```sh
   conda install openvino-ie4py -c intel
   ```
4. Verify the package installed:
   ```sh
   python -c "from openvino.inference_engine import IECore"
   ```
   
Now you can start to develop and run your application.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit).
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org).
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- Intel® Distribution of OpenVINO™ toolkit Anaconda* home page: [https://anaconda.org/intel/openvino-ie4py](https://anaconda.org/intel/openvino-ie4py)

