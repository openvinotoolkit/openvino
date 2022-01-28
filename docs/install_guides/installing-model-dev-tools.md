# Install OpenVINO Model Development Tools {#openvino_docs_install_guides_install_dev_tools}

If you are planning to develop your own inference solutions, install OpenVINO Model Development Tools, which provides the following tools:

* Model Optimizer
* Benchmark Tool
* Accuracy Checker and Annotation Converter
* Post-Training Optimization Tool
* Model Downloader and other Open Model Zoo tools

From 2022.1 release, the OpenVINO Developer Tools can only be installed via PyPI.

## For C++ developers

You can install OpenVINO Model Development Tools via either of the following ways:

* **Recommended**: Install using the requirements files

    1. If you have installed OpenVINO Runtime from an installer, APT or YUM repository, you can find a set of requirements files in `<installdir>\tools\` directory.
    2. You can manually install the same version of OpenVINO Model Development Tools by using the requirements files. To install mandatory requirements only, use the following command:
    ```
    pip install -r <installdir>\tools\requirements.txt
    ```
    3. Make sure that you also install your additional frameworks with the corresponding requirements files. For example, if you are using a TensorFlow model, use the following command to install requirements for TensorFlow:  
    ```
    pip install -r <installdir>\tools\requirements_tensorflow2.txt
    ```
* Install from the openvino-dev package

    You can use the following command to install the latest package version available in the index:
    ```
    pip install openvino-dev[EXTRAS]
    ```
    where the EXTRAS parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.
    For more details, see <https://pypi.org/project/openvino-dev/>.

    > **NOTE**:
    > * If you have installed OpenVINO Runtime via the installer, to avoid version conflicts, specify your version in the command:
    ```sh
    pip install openvino-dev[EXTRAS]==<version>
    ```
    > * For TensorFlow, use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.
    
    
    > **NOTE**:
    > * If you have installed OpenVINO Runtime via the installer, to avoid version conflicts, specify your version in the command:
    
    ```sh
    pip install openvino-dev[EXTRAS]==<version>
    ```
    
    > * For TensorFlow, use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.
    

## For Python developers

You can use the following command to install the latest package version available in the index:
```
pip install openvino-dev[EXTRAS]
```
where the EXTRAS parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

> **NOTE**: For TensorFlow, use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.

For more details, see <https://pypi.org/project/openvino-dev/>.
