# Install OpenVINO™ Runtime on Windows from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_windows}

With the OpenVINO™ 2022.2 release, you can download and use archive files to install OpenVINO Runtime. The archive files contain pre-built binaries and library files needed for OpenVINO Runtime, as well as sample code for running demos. This page provides instructions showing how to install OpenVINO Runtime using archive files. Check the [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes) for more information on updates in the 2022.2 release.

Installing OpenVINO Runtime from archive files is recommended for C++ developers. If you are working with Python, the PyPI package has everything needed for Python development and deployment on CPU and GPUs. Visit the [Install OpenVINO from PyPI](installing-openvino-pip.md) page for instructions on how to install OpenVINO Runtime for Python using PyPI.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  * Microsoft Windows 10, 64-bit
  * Microsoft Windows 11, 64-bit

.. tab:: Hardware

  Optimized for these processors:

  * 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors 
  * 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
  * Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
  * Intel Atom® processor with support for Intel® Streaming SIMD Extensions 4.1 (Intel® SSE4.1)
  * Intel Pentium® processor N4200/5, N3350/5, or N3450/5 with Intel® HD Graphics
  * Intel® Iris® Xe MAX Graphics
  * Intel® Neural Compute Stick 2
  * Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
  
.. tab:: Processor Notes

  Processor graphics are not included in all processors. 
  See `Product Specifications`_ for information about your processor.
  
  .. _Product Specifications: https://ark.intel.com/

.. tab:: Software

  * `Microsoft Visual Studio 2019 with MSBuild <http://visualstudio.microsoft.com/downloads/>`_
  * `CMake 3.14 or higher, 64-bit <https://cmake.org/download/>`_ (optional, only required for building sample applications)
  * `Python 3.6 - 3.10, 64-bit <https://www.python.org/downloads/windows/>`_
     * Note that OpenVINO is gradually phasing out support for Python 3.6. Python 3.7 - 3.10 are recommended.

  .. note::
     To install Microsoft Visual Studio 2019, follow the `Microsoft Visual Studio installation guide <https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019>`_. You can choose to download the Community version. During installation in the **Workloads** tab, choose **Desktop development with C++**.

  .. note::
    You can either use `cmake<version>.msi` which is the installation wizard or `cmake<version>.zip` where you have to go into the `bin` folder and then manually add the path to environmental variables.
  
  .. important::
    When installing Python, make sure you click the option **Add Python 3.x to PATH** to `add Python <https://docs.python.org/3/using/windows.html#installation-steps>`_ to your `PATH` environment variable.

@endsphinxdirective

## Installing OpenVINO Runtime

### <a name="install-openvino"></a>Step 1: Download and Install OpenVINO Core Components

1. Open a command terminal as administrator by searching for "Command Terminal" in the Start menu, right clicking on it, and selecting "Run as administrator". Create a folder for OpenVINO and move into it by issuing the following commands.

   ```sh
   mkdir "C:\Program Files (x86)\Intel"
   cd "C:\Program Files (x86)\Intel"
   ```

   > **NOTE**: C:\Program Files (x86)\Intel is the recommended folder path, but you may use a different path if desired or if you don't have administrator priveleges on your PC. If the C:\Program Files (x86)\Intel folder already exists, skip the `mkdir` command.

2. Download the OpenVINO Runtime 2022.2 archive file from the [OpenVINO releases page](https://github.com/openvinotoolkit/openvino/releases/tag/2022.2.0), unzip it, and rename the folder to `openvino_2022.2.0.7713` by issuing:
   ```sh
   curl -L https://github.com/openvinotoolkit/openvino/releases/download/2022.2.0/w_openvino_toolkit_windows_2022.2.0.7713.af16ea1d79a_x86_64.zip --output openvino_2022.2.0.7713.zip
   tar -xf openvino_2022.2.0.7713.zip
   ren w_openvino_toolkit_windows_2022.2.0.7713.af16ea1d79a_x86_64 openvino_2022.2.0.7713
   ```

3. Create a symbolic link to the folder by issuing:
   ```sh
   mklink /D openvino_2022 openvino_2022.2.0.7713
   ```

   > **NOTE**: If you have already installed a previous release of OpenVINO 2022, a symbolic link to the `openvino_2022` folder may already exist. If you want to override it, nagivate to the `C:\Program Files (x86)\Intel` folder and delete the existing linked folder before running the `mklink` command.

Congratulations, you finished installation! The C:\Program Files (x86)\Intel\openvino_2022 folder now contains the core components for OpenVINO™. When other pages in OpenVINO™ documentation refer to the `<INSTALL_DIR>` directory, this is the folder they're referring to. If you installed OpenVINO™ in a different location, make sure to use that instead.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Open the Command Prompt, and run the `setupvars.bat` batch file to temporarily set your environment variables. Again, if you installed OpenVINO™ in a folder other than `C:\Program Files (x86)\Intel\openvino_2022`, use that location instead.

```sh
"C:\Program Files (x86)\Intel\openvino_2022\setupvars.bat"
```

> **Important**: The above command must be re-run every time a new Command Prompt window is opened.

> **NOTE**: If you see an error indicating Python is not installed, Python may not be added to the PATH environment variable (as described [here](https://docs.python.org/3/using/windows.html#finding-the-python-executable)). Check your system environment variables, and add Python if necessary.

The environment variables are set. Continue to the next section if you want to download any additional components.

### <a name="model-optimizer">Step 3 (Optional): Install Additional Components</a>

OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Runtime using archive files, OpenVINO Development Tools must be installed separately.

See the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page for step-by-step installation instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on Github](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

### <a name="optional-steps"></a>Step 4 (Optional): Configure Inference on non-CPU Devices
OpenVINO Runtime has a plugin architecture that enables you to run inference on multiple devices without rewriting your code. Supported devices include integrated GPUs, discrete GPUs, NCS2, VPUs, and GNAs. See the instructions below to set up OpenVINO on these devices.

@sphinxdirective
.. tab:: GPU

   To enable the toolkit components to use processor graphics (GPU) on your system, follow the steps in :ref:`GPU Setup Guide <gpu guide windows>`.

.. tab:: VPU

   To install and configure your Intel® Vision Accelerator Design with Intel® Movidius™ VPUs, see the :ref:`VPU Configuration Guide <vpu guide windows>`.
   
.. tab:: NCS 2
   
   No additional configurations are needed.
   
.. tab:: GNA

   To enable the toolkit components to use Intel® Gaussian & Neural Accelerator (GNA) on your system, follow the steps in :ref:`GNA Setup Guide <gna guide windows>`.
   
@endsphinxdirective

## <a name="get-started"></a>What's Next?
Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

### Get started with Python
<img src="https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif" width=400>

Try the [Python Quick Start Example](https://docs.openvino.ai/2022.2/notebooks/201-vision-monodepth-with-output.html) to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside your web browser.

Visit the [Tutorials](../tutorials.md) page for more Jupyter Notebooks to get you started with OpenVINO, such as:
* [OpenVINO Python API Tutorial](https://docs.openvino.ai/2022.2/notebooks/002-openvino-api-with-output.html)
* [Basic image classification program with Hello Image Classification](https://docs.openvino.ai/2022.2/notebooks/001-hello-world-with-output.html)
* [Convert a PyTorch model and use it for image background removal](https://docs.openvino.ai/2022.2/notebooks/205-vision-background-removal-with-output.html)

### Get started with C++
<img src="https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg" width=400>

Try the [C++ Quick Start Example](@ref openvino_docs_get_started_get_started_demos) for step-by-step instructions on building and running a basic image classification C++ application.

Visit the [Samples](../OV_Runtime_UG/Samples_Overview.md) page for other C++ example applications to get you started with OpenVINO, such as:
* [Basic object detection with the Hello Reshape SSD C++ sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)
* [Automatic speech recognition C++ sample](@ref openvino_inference_engine_samples_speech_sample_README)

## <a name="uninstall"></a>Uninstalling the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

@sphinxdirective

.. dropdown:: Additional Resources
      
   * Converting models for use with OpenVINO™: :ref:`Model Optimizer Developer Guide <deep learning model optimizer>`
   * Writing your own OpenVINO™ applications: :ref:`OpenVINO™ Runtime User Guide <deep learning openvino runtime>`
   * Sample applications: :ref:`OpenVINO™ Toolkit Samples Overview <code samples>`
   * Pre-trained deep learning models: :ref:`Overview of OpenVINO™ Toolkit Pre-Trained Models <model zoo>`
   * IoT libraries and code samples in the GitHUB repository: `Intel® IoT Developer Kit`_ 
      
<!---  
   To learn more about converting models from specific frameworks, go to: 
   * :ref:`Convert Your Caffe Model <convert model caffe>`
   * :ref:`Convert Your TensorFlow Model <convert model tf>`
   * :ref:`Convert Your Apache MXNet Model <convert model mxnet>`
   * :ref:`Convert Your Kaldi Model <convert model kaldi>`
   * :ref:`Convert Your ONNX Model <convert model onnx>`
--->    
   .. _Intel® IoT Developer Kit: https://github.com/intel-iot-devkit

@endsphinxdirective
