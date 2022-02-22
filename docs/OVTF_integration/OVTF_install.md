# Installation using Pre-Built Packages {#ovtf_install}

## Package Overview

**OpenVINO™ integration with TensorFlow** is released for Linux, macOS, and Windows. You can choose one of the following methods based on your requirements.

### Linux

  **OpenVINO™ integration with TensorFlow** on Linux is released in two different versions: 

- **PyPi release alongside PyPi TensorFlow**  
   - Built with CXX11_ABI=0  
   - Includes OpenVINO™ pre-built libraries (no need to install OpenVINO™ separately)
   - Supports Intel® CPU, iGPU, and MYRIAD technologies (no VAD-M support)

- **Package alongside the Intel® Distribution of OpenVINO™ Toolkit**  
   - Built with CXX11_ABI=1
   - Compatible with OpenVINO™ version 2021.4.2
   - Supports Intel® CPU, iGPU, MYRIAD, and VAD-M technologies
   - Needs a custom TensorFlow ABI1 package, which is available in Github release

### macOS

- [OpenVINO™ integration with TensorFlow PyPi release alongside PyPi TensorFlow](#InstallOpenVINOintegrationwithTensorFlowalongsidePyPiTensorFlow)

     - Includes pre-built libraries of OpenVINO™ version 2021.4.2. The users do not have to install OpenVINO™ separately 
     - Supports Intel® CPUs, Intel® integrated GPUs, and Intel® Movidius™ Vision Processing Units (VPUs). No VAD-M support

### Windows

- [OpenVINO™ integration with TensorFlow PyPi release alongside TensorFlow released in Github](#InstallOpenVINOintegrationwithTensorFlowalongsideTensorFlow)
     
     - Includes pre-built libraries of OpenVINO™ version 2021.4.2. The users do not have to install OpenVINO™ separately 
     - Supports Intel® CPUs, Intel® integrated GPUs, and Intel® Movidius™ Vision Processing Units (VPUs). No VAD-M support
     - TensorFlow wheel for Windows from PyPi does't have all the API symbols enabled which are required for OpenVINO™ integration with TensorFlow. User needs to install the  TensorFlow wheel from the assets of the Github release page.


### summary
@sphinxdirective 

+-------------------------------+------------------------------+-------------------------------+
|| **TensorFlow Pip Package**   || tensorflow                  || tensorflow-abi1              |
+-------------------------------+------------------------------+-------------------------------+
|| OpenVINO™ integration        || openvino-tensorflow         || openvino-tensorflow-abi1     |
|| with TensorFlow Pip Package  |                              |                               |
+-------------------------------+------------------------------+-------------------------------+
|| Supported OpenVINO™ Flavor   || OpenVINO™ built from source || Dynamically links to         |
|                               |                              || OpenVINO™ binary release     |
+-------------------------------+------------------------------+-------------------------------+
|| Supported Hardware           || CPU,iGPU,MYRIAD             || CPU,iGPU,MYRIAD,VAD-M        |
+-------------------------------+------------------------------+-------------------------------+
|| Comments                     || OpenVINO™ libraries are     || OpenVINO™ integration with   |
|                               | built from source and        | TensorFlow libraries are      |
|                               | included in the wheel package| dynamically linked to         |
|                               |                              | OpenVINO™ binaries            |
+-------------------------------+------------------------------+-------------------------------+

@endsphinxdirective 

## Installation

### Alongside PyPi TensorFlow (works on Linux and macOS)

```bash
pip3 install -U pip
pip3 install tensorflow==2.7.0
pip3 install -U openvino-tensorflow
```

### Alongside TensorFlow released on Github (Works on Windows)

```bash
pip3.9 install -U pip
pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0tensorflow-2.      70-cp39-cp39-win_amd64.whl
pip3.9 install -U openvino-tensorflow
```


### Alongside the Intel® Distribution of OpenVINO™ Toolkit (Works on Linux)

1. Ensure the following versions are being used for pip and numpy:

```bash
pip3 install -U pip
pip3 install numpy==1.20.2
```

2. Install `TensorFlow` based on your Python version. You can [build TensorFlow from source](https://github.com/openvinotoolkit/openvino_tensorflow/blob/master/docs/BUILD.md#tensorflow) with **-D_GLIBCXX_USE_CXX11_ABI=1** or follow the instructions below to use the appropriate package:

```bash
pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0tensorflow_abi1-2.7.0-cp37-cp37m-manylinux2010_x86_64.whl
or
pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0tensorflow_abi1-2.7.0-cp38-cp38-manylinux2010_x86_64.whl
or
pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0tensorflow_abi1-2.7.0-cp39-cp39-manylinux2010_x86_64.whl
```

3. Download & install Intel® Distribution of OpenVINO™ Toolkit 2021.4.2 release along with its dependencies from ([https://software.intel.com/en-us/openvino-toolkit/download](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html)).

4. Initialize the OpenVINO™ environment by running the `setupvars.sh` located in 
`openvino_install_directory/bin`:

```bash
source setupvars.sh
```

5. Install `openvino-tensorflow`. Based on your Python version, choose the appropriate package below:

```bash
pip3.7 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0openvino_tensorflow_abi1-1.1.0-cp37-cp37m-linux_x86_64.whl
or
pip3.8 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0openvino_tensorflow_abi1-1.1.0-cp38-cp38-linux_x86_64.whl
or
pip3.9 install https://github.com/openvinotoolkit/openvino_tensorflow/releases/download/v1.1.0openvino_tensorflow_abi1-1.1.0-cp39-cp39-linux_x86_64.whl
```
