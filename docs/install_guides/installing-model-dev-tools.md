# Install OpenVINO™ Development Tools {#openvino_docs_install_guides_install_dev_tools}

OpenVINO Development Tools is a set of utilities that make it easy to develop and optimize models and applications for OpenVINO. It provides the following tools:

* Model Optimizer
* Benchmark Tool
* Accuracy Checker and Annotation Converter
* Post-Training Optimization Tool
* Model Downloader and other Open Model Zoo tools

The instructions on this page show how to install OpenVINO Development Tools. If you are a Python developer, it only takes a few simple steps to install the tools with PyPI: see the <a href="openvino_docs_install_guides_install_dev_tools.html#python-developers">For Python Developers</a> section for more information. If you are developing in C++, OpenVINO Runtime must be installed separately before installing OpenVINO Development Tools: see the <a href="openvino_docs_install_guides_install_dev_tools.html#cpp-developers">For C++ Developers</a> section for more information. In both cases, you will need to have Python 3.6 - 3.9 installed on your machine before starting.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. 

### <a name="python-developers"></a>For Python Developers

If you are a Python developer, follow the steps in the <a href="openvino_docs_install_guides_install_dev_tools.html#install-dev-tools">Installing OpenVINO Development Tools</a> section on this page to install it. Installing OpenVINO Development Tools will also install OpenVINO Runtime as a dependency, so you don’t need to install OpenVINO Runtime separately.
   
### <a name="cpp-developers"></a>For C++ Developers
If you are a C++ developer, you must first install OpenVINO Runtime separately to set up the C++ libraries, sample code, and dependencies for building applications with OpenVINO. These files are not included with the PyPI distribution. Visit the <a href="https://docs.openvino.ai/latest/openvino_docs_install_guides_install_runtime.html">Install OpenVINO Runtime</a> page, select your operating system, and select the option to install OpenVINO Runtime from an archive file.


## <a name="install-dev-tools"></a>Installing OpenVINO™ Development Tools

### Step 1. Set Up Python Virtual Environment

Use a virtual environment to avoid dependency conflicts. 

To create a virtual environment, use the following command:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      python3 -m venv openvino_env
   
.. tab:: Windows

   .. code-block:: sh
   
      python -m venv openvino_env
     
     
@endsphinxdirective


### Step 2. Activate Virtual Environment

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      source openvino_env/bin/activate
   
.. tab:: Windows

   .. code-block:: sh
   
      openvino_env\Scripts\activate
     
     
@endsphinxdirective


### Step 3. Set Up and Update PIP to the Highest Version

Use the following command:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

To install and configure the components of the development package for working with specific frameworks, use the following command:
```
pip install openvino-dev[extras]
```
where the `extras` parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

For example, to install and configure the components for working with TensorFlow 2.x and ONNX, use the following command:
```
pip install openvino-dev[tensorflow2,onnx]
```

> **NOTE**: Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use TensorFlow 2.x environment to convert both TensorFlow 1.x and 2.x models. Use the `tensorflow2` value as much as possible. The `tensorflow` value is provided only for compatibility reasons.


### Step 5. Verify the Installation

To verify if the package is properly installed, run the command below (this may take a few seconds):
```sh
mo -h
```
You will see the help message for Model Optimizer if installation finished successfully.


## For C++ Developers

Note the following things:

* To install OpenVINO Development Tools, you must have OpenVINO Runtime installed first. You can install OpenVINO Runtime through archive files. See [Install OpenVINO on Linux from Archive](installing-openvino-from-archive-linux.md), [Install OpenVINO on Windows from Archive](installing-openvino-from-archive-windows.md), and [Install OpenVINO on macOS from Archive](installing-openvino-from-archive-macos.md) for more details. 
* Ensure that the version of OpenVINO Development Tools you are installing matches that of OpenVINO Runtime. 

Use either of the following ways to install OpenVINO Development Tools:

### Recommended: Install Using the Requirements Files

1. After you have installed OpenVINO Runtime from an archive file, you can find a set of requirements files in the `<INSTALL_DIR>\tools\` directory. Select the most suitable ones to use.
2. Install the same version of OpenVINO Development Tools by using the requirements files. 
   To install mandatory requirements only, use the following command:
   ```
   pip install -r <INSTALL_DIR>\tools\requirements.txt
   ```
3. Make sure that you also install your additional frameworks with the corresponding requirements files. For example, if you are using a TensorFlow model, use the following command to install requirements for TensorFlow:  
```
pip install -r <INSTALL_DIR>\tools\requirements_tensorflow2.txt
```

### Alternative: Install from the openvino-dev Package

You can also use the following command to install the latest package version available in the index:
```
pip install openvino-dev[EXTRAS]
```
where the EXTRAS parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

If you have installed OpenVINO Runtime via the installer, to avoid version conflicts, specify your version in the command. For example:
```
pip install openvino-dev[tensorflow2,onnx]==2022.1
```
    
> **NOTE**: Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use TensorFlow 2.x environment to convert both TensorFlow 1.x and 2.x models. The `tensorflow` value is provided only for compatibility reasons, use the `tensorflow2` value instead.

For more details, see <https://pypi.org/project/openvino-dev/>.

## What's Next?

Now you may continue with the following tasks:

* To convert models for use with OpenVINO, see [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
* See pre-trained deep learning models in our [Open Model Zoo](../model_zoo.md).
* Try out OpenVINO via [OpenVINO Notebooks](https://docs.openvino.ai/latest/notebooks/notebooks.html).
* To write your own OpenVINO™ applications, see [OpenVINO Runtime User Guide](../OV_Runtime_UG/openvino_intro.md).
* See sample applications in [OpenVINO™ Toolkit Samples Overview](../OV_Runtime_UG/Samples_Overview.md).

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
