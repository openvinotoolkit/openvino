# Install OpenVINO™ Runtime on Windows 10 from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_windows}

With the OpenVINO™ 2022.2 release, you can download and use archive files to install OpenVINO Runtime.

You can also check the [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes) for more information on updates in this release.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  Microsoft Windows 10, 64-bit

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

  * For C++ developers:
     * `Microsoft Visual Studio 2019 with MSBuild <http://visualstudio.microsoft.com/downloads/>`_
     * `CMake 3.14 or higher, 64-bit <https://cmake.org/download/>`_ (optional, only required for building sample applications)
  * For Python developers: `Python 3.6 - 3.9, 64-bit <https://www.python.org/downloads/windows/>`_
     * Note that OpenVINO is gradually stopping the support for Python 3.6. Python 3.7 - 3.9 are recommended.

  .. note::
    You can choose to download Community version. Use `Microsoft Visual Studio installation guide <https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2019>`_ to walk you through the installation. During installation in the **Workloads** tab, choose **Desktop development with C++**.

  .. note::
    You can either use `cmake<version>.msi` which is the installation wizard or `cmake<version>.zip` where you have to go into the `bin` folder and then manually add the path to environmental variables.
  
  .. important::
    As part of this installation, make sure you click the option **Add Python 3.x to PATH** to `add Python <https://docs.python.org/3/using/windows.html#installation-steps>`_ to your `PATH` environment variable.

@endsphinxdirective

## Installing OpenVINO Runtime

### <a name="install-openvino"></a>Step 1: Download and Install OpenVINO Core Components

1. Select and download the OpenVINO™ archive files from [Intel® Distribution of OpenVINO™ toolkit download page](https://software.intel.com/en-us/openvino-toolkit/choose-download).
   There are typically two files for you to download: 
   ```sh
   w_openvino_toolkit_<operating system>_<release version>_<package ID>_x86_64.zip
   w_openvino_toolkit_<operating system>_<release version>_<package ID>_x86_64.zip.sha256
   ``` 
   where the `.sha256` file is used to verify the success of the download process.
   
2. Locate the downloaded files in your system. This document assumes the files are in your `Downloads` directory. 
   
3. Open a command prompt terminal window, and run the following command with the `.zip` file:
   ```sh
   CertUtil -hashfile <archive name>.zip SHA256
   ```
   Compare the returned value in the output with what's in the `<archive name>.zip.sha256` file:
   * If the values are the same, you have downloaded the correct file successfully.
   * If not, create a Support ticket [here](https://www.intel.com/content/www/us/en/support/contact-intel.html).

4. Unzip the `<archive name>.zip` file using your preferred archive tool. 
   > **NOTE**: The name of the archive file might be long. Make sure that you have [enabled long paths on Windows](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later) to avoid possible errors in the unzipping process.

 
The standard OpenVINO `<INSTALL_DIR>` used in this document is `C:\Program Files (x86)\Intel\openvino_<version>\`. You're recommended to move the extracted files to that directory. 

For simplicity, you can also create a symbolic link to the latest installation: `C:\Program Files (x86)\Intel\openvino_2022\`.

The core components are now installed. Continue to the next section to configure the environment.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

> **NOTE**: If you installed the Intel® Distribution of OpenVINO™ to a non-default install directory, replace `C:\Program Files (x86)\Intel` with that directory in this guide's instructions.

You must update several environment variables before you can compile and run OpenVINO™ applications. Open the Command Prompt, and run the `setupvars.bat` batch file to temporarily set your environment variables:

```sh
"<INSTALL_DIR>\setupvars.bat"
```

**Optional**: OpenVINO™ toolkit environment variables are removed when you close the command prompt window. You can permanently set the environment variables manually.

> **NOTE**: If you see an error indicating Python is not installed when you know you installed it, your computer might not be able to find the program. Check your system environment variables, and add Python if necessary.

The environment variables are set. Next, you can download some additional tools.

### <a name="model-optimizer">Step 3 (Optional): Install Additional Components</a>

Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter are not part of the installer. The OpenVINO™ Development Tools can only be installed via PyPI now. See [Install OpenVINO™ Development Tools](installing-model-dev-tools.md) for detailed steps.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on Github](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

### <a name="optional-steps"></a>Step 4 (Optional): Configure Inference on non-CPU Devices

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

Now you are ready to try out the toolkit.

Start with some Python tutorials:
   * [Hello Image Classification](https://docs.openvino.ai/latest/notebooks/001-hello-world-with-output.html)
   * [Convert TensorFlow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Convert a PyTorch model and remove the image background](https://docs.openvino.ai/latest/notebooks/205-vision-background-removal-with-output.html)

To start with C++ samples, see <a href="openvino_docs_OV_UG_Samples_Overview.html#build-samples-windows">Build Sample Applications on Windows</a> first, and then you can try the following samples:
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)
    
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
