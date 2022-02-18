# Install OpenVINO™ Development Tools {#openvino_docs_install_guides_install_dev_tools}

If you are planning to develop your own inference solutions, install OpenVINO™ Development Tools, which provides the following tools:

* Model Optimizer
* Benchmark Tool
* Accuracy Checker and Annotation Converter
* Post-Training Optimization Tool
* Model Downloader and other Open Model Zoo tools

From the 2022.1 release, OpenVINO Development Tools can only be installed via PyPI.


## For C++ Developers

Note the following things:

* To install OpenVINO Development Tools, you must have OpenVINO Runtime installed first. You can install OpenVINO Runtime through an installer ([Linux](installing-openvino-linux.md), [Windows](installing-openvino-windows.md), or [macOS](installing-openvino-macos.md)), [APT for Linux](installing-openvino-apt.md) or [YUM for Linux](installing-openvino-yum.md). 
* Ensure that the version of OpenVINO Development Tools you are installing matches that of OpenVINO Runtime. 

Use either of the following ways to install OpenVINO Development Tools:

### Recommended: Install Using the Requirements Files

1. After you have installed OpenVINO Runtime from an installer, APT or YUM repository, you can find a set of requirements files in the `<INSTALLDIR>\tools\` directory. Select the most suitable ones to use.
2. Install the same version of OpenVINO Development Tools by using the requirements files. 
   To install mandatory requirements only, use the following command:
   ```
   pip install -r <INSTALLDIR>\tools\requirements.txt
   ```
3. Make sure that you also install your additional frameworks with the corresponding requirements files. For example, if you are using a TensorFlow model, use the following command to install requirements for TensorFlow:  
```
pip install -r <INSTALLDIR>\tools\requirements_tensorflow2.txt
```

### Alternative: Install from the openvino-dev Package

You can also use the following command to install the latest package version available in the index:
```
pip install openvino-dev[EXTRAS]
```
where the EXTRAS parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

If you have installed OpenVINO Runtime via the installer, to avoid version conflicts, specify your version in the command. For example:
```
pip install openvino-dev[tensorflow2,mxnet,caffe]==2022.1
```
    
> **NOTE**: For TensorFlow, use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.

For more details, see <https://pypi.org/project/openvino-dev/>.

    
## For Python Developers

You can use the following command to install the latest package version available in the index:
```
pip install openvino-dev[EXTRAS]
```
where the EXTRAS parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

For example, to install and configure the components for working with TensorFlow 2.x, MXNet and Caffe, use the following command:
```
pip install openvino-dev[tensorflow2,mxnet,caffe]
```

> **NOTE**: For TensorFlow, use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.

For more details, see <https://pypi.org/project/openvino-dev/>.
