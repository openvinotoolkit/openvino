# Install OpenVINO™ Development Tools {#openvino_docs_install_guides_install_dev_tools}

OpenVINO Development Tools is a set of utilities that make it easy to develop and optimize models and applications for OpenVINO. It provides the following tools:

* Model Optimizer
* Benchmark Tool
* Accuracy Checker and Annotation Converter
* Post-Training Optimization Tool
* Model Downloader and other Open Model Zoo tools

The instructions on this page show how to install OpenVINO Development Tools. If you are a Python developer, it only takes a few simple steps to install the tools with PyPI. If you are developing in C++, OpenVINO Runtime must be installed separately before installing OpenVINO Development Tools.

In both cases, Python 3.7 - 3.10 needs to be installed on your machine before starting.

> **NOTE**: From the 2022.1 release, the OpenVINO™ Development Tools can only be installed via PyPI. 

## <a name="python-developers"></a>For Python Developers

If you are a Python developer, follow the steps in the <a href="openvino_docs_install_guides_install_dev_tools.html#install-dev-tools">Installing OpenVINO Development Tools</a> section on this page to install it. Installing OpenVINO Development Tools will also install OpenVINO Runtime as a dependency, so you don’t need to install OpenVINO Runtime separately. This option is recommended for new users.
   
## <a name="cpp-developers"></a>For C++ Developers
If you are a C++ developer, you must first install OpenVINO Runtime separately to set up the C++ libraries, sample code, and dependencies for building applications with OpenVINO. These files are not included with the PyPI distribution. See the [Install OpenVINO Runtime](./installing-openvino-runtime.md) page to install OpenVINO Runtime from an archive file for your operating system.

Once OpenVINO Runtime is installed, you may install OpenVINO Development Tools for access to tools like Model Optimizer, Model Downloader, Benchmark Tool, and other utilities that will help you optimize your model and develop your application. Follow the steps in the <a href="#install-dev-tools">Installing OpenVINO Development Tools</a> section on this page to install it. In Step 4, make sure that you follow the instructions in the "C++" tab.

## <a name="install-dev-tools"></a>Installing OpenVINO™ Development Tools
Follow these step-by-step instructions to install OpenVINO Development Tools on your computer.

### Step 1. Set Up Python Virtual Environment

Create a virtual Python environment to avoid dependency conflicts. To create a virtual environment, use the following command:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh
   
      python3 -m venv openvino_env
   
.. tab:: Windows

   .. code-block:: sh
   
      python -m venv openvino_env
     
     
@endsphinxdirective


### Step 2. Activate Virtual Environment

Activate the newly created Python virtual environment by issuing this command:

@sphinxdirective

.. tab:: Linux and macOS

   .. code-block:: sh

      source openvino_env/bin/activate

.. tab:: Windows

   .. code-block:: sh

      openvino_env\Scripts\activate

.. important::

   The above command must be re-run every time a new command terminal window is opened.

@endsphinxdirective


### Step 3. Set Up and Update PIP to the Highest Version
Make sure `pip` is installed in your environment and upgrade it to the latest version by issuing the following command:

```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

To install and configure the components of the development package for working with specific frameworks, use the commands below.

Note that the commands are different for a Python installation and a C++ installation. If you want to develop with Python only, follow the instructions in the **Python** tab. If you want to develop with C++, first make sure you have installed OpenVINO Runtime using archive files as stated in the <a href="#cpp-developers">For C++ Developers</a> section, then follow the instructions in the **C++** tab.


@sphinxdirective

.. tab:: Python

   To install OpenVINO Development Tools into the existing environment with the deep learning framework of your choice, run the following command:

   .. code-block:: sh

      pip install openvino-dev

   In case that you encounter any compatibility issues between OpenVINO and your deep learning framework, you may install OpenVINO Development Tools into a separate environment. Use the following command to get specific validated versions of your framework:

   .. code-block:: sh

      pip install openvino-dev[extras]

   where the `extras` parameter specifies one or more deep learning frameworks via these values: `caffe`, `kaldi`, `mxnet`, `onnx`, `pytorch`, `tensorflow`, `tensorflow2`. Make sure that you install the corresponding frameworks for your models.

   For example, to install and configure the components for working with TensorFlow 2.x and ONNX, use the following command:

   .. code-block:: sh

      pip install openvino-dev[tensorflow2,onnx]

   .. note::

      Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use the `tensorflow2` parameter to install a TensorFlow 2.x environment that can convert both TensorFlow 1.x and 2.x models. If your model isn't compatible with the TensorFlow 2.x environment, use the `tensorflow` parameter to install the TensorFlow 1.x environment. The TF 1.x environment is provided only for legacy compatibility reasons.

.. tab:: C++

   When using OpenVINO Development Tools for C++ development, it’s important to install the same version as OpenVINO Runtime. Following the instructions below will ensure that you are installing a version that matches that of OpenVINO Runtime.

   **Recommended: Install Using the Requirements Files**

   After you have installed OpenVINO Runtime from an archive file, you can find a set of requirements files in the <INSTALL_DIR>\tools\ directory. The requirements files will install the matching version of OpenVINO Development Tools and its dependencies.

   1. Install the OpenVINO Development Tools mandatory requirements using the following command:

      .. code-block:: sh

         pip install -r <INSTALL_DIR>\tools\requirements.txt

   2. If you are using additional frameworks, you must also install the requirements for those frameworks using the corresponding requirements file. For example, if you are using a TensorFlow model, use the following command to install requirements for TensorFlow:

      .. code-block:: sh

         pip install -r <INSTALL_DIR>\tools\requirements_tensorflow2.txt

   **Alternative: Install the openvino-dev Package from PyPI**

   You can also install OpenVINO Development Tools from PyPI using the following command.

   .. important::

      Make sure that the `openvino-dev` version you specified matches your installed version of OpenVINO Runtime. Otherwise, compatibility errors are likely to occur.

   .. code-block:: sh

      pip install openvino-dev[EXTRAS]==2022.3.0

   where the EXTRAS parameter specifies one or more deep learning frameworks via these values: caffe, kaldi, mxnet, onnx, pytorch, tensorflow, tensorflow2. Make sure that you install the corresponding frameworks for your models. For example: 

   .. code-block:: sh

      pip install openvino-dev[tensorflow2,onnx]==2022.3.0

   .. note::

      Model Optimizer support for TensorFlow 1.x environment has been deprecated. Use the `tensorflow2` parameter or to install a TensorFlow 2.x environment that can convert both TensorFlow 1.x and 2.x models. If your model isn't compatible with the TensorFlow 2.x environment, use the `tensorflow` parameter to install the TensorFlow 1.x environment. The TF 1.x environment is provided only for legacy compatibility reasons.

@endsphinxdirective


For more details on the openvino-dev PyPI package, see https://pypi.org/project/openvino-dev/.

### Step 4. Test the Installation

To verify the package is properly installed, run the command below (this may take a few seconds):

```sh
mo -h
```

You will see the help message for Model Optimizer if installation finished successfully. If you get an error, refer to the [Troubleshooting Guide](./troubleshooting.md) for possible solutions.

Congratulations! You finished installing OpenVINO Development Tools with C++ capability. Now you can start exploring OpenVINO's functionality through example C++ applications. See the "What's Next?" section to learn more!

## What's Next?
Learn more about OpenVINO and use it in your own application by trying out some of these examples!

### Get started with Python
<img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=400>

Try the [Python Quick Start Example](https://docs.openvino.ai/nightly/notebooks/201-vision-monodepth-with-output.html) to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

Visit the [Tutorials](../tutorials.md) page for more Jupyter Notebooks to get you started with OpenVINO, such as:
* [OpenVINO Python API Tutorial](https://docs.openvino.ai/nightly/notebooks/002-openvino-api-with-output.html)
* [Basic image classification program with Hello Image Classification](https://docs.openvino.ai/nightly/notebooks/001-hello-world-with-output.html)
* [Convert a PyTorch model and use it for image background removal](https://docs.openvino.ai/nightly/notebooks/205-vision-background-removal-with-output.html)

### Get started with C++
<img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=400>

Try the [C++ Quick Start Example](@ref openvino_docs_get_started_get_started_demos) for step-by-step instructions on building and running a basic image classification C++ application.

Visit the [Samples](../OV_Runtime_UG/Samples_Overview.md) page for other C++ example applications to get you started with OpenVINO, such as:
* [Basic object detection with the Hello Reshape SSD C++ sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)
* [Automatic speech recognition C++ sample](@ref openvino_inference_engine_samples_speech_sample_README)

### Learn OpenVINO Development Tools
* Explore a variety of pre-trained deep learning models in the <a href="model_zoo.html">Open Model Zoo</a> and deploy them in demo applications to see how they work.
* Want to import a model from another framework and optimize its performance with OpenVINO? Visit the <a href="openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html">Model Optimizer Developer Guide</a>.
* Accelerate your model's speed even further with quantization and other compression techniques using <a href="pot_introduction.html">Post-Training Optimization Tool</a>.
* Benchmark your model's inference speed with one simple command using the <a href="openvino_inference_engine_tools_benchmark_tool_README.html">Benchmark Tool</a>.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: <https://software.intel.com/en-us/openvino-toolkit>
- For IoT Libraries & Code Samples, see [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).
