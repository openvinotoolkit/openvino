# Install OpenVINO™ Runtime on macOS from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_macos}

With the OpenVINO™ 2022.2 release, you can download and use archive files to install OpenVINO Runtime.

You can also check the [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes) for more information on updates in this release.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

> **NOTE**: The Intel® Distribution of OpenVINO™ toolkit is supported on macOS version 10.15 with Intel® processor-based machines.

## System Requirements

@sphinxdirective
.. tab:: Operating Systems

  macOS 10.15

.. tab:: Hardware

  Optimized for these processors:

  * 6th to 12th generation Intel® Core™ processors and Intel® Xeon® processors 
  * 3rd generation Intel® Xeon® Scalable processor (formerly code named Cooper Lake)
  * Intel® Xeon® Scalable processor (formerly Skylake and Cascade Lake)
  * Intel® Neural Compute Stick 2
  
  .. note::
    The current version of the Intel® Distribution of OpenVINO™ toolkit for macOS supports inference on Intel CPUs and Intel® Neural Compute Stick 2 devices only.

.. tab:: Software Requirements

  * `CMake 3.13 or higher <https://cmake.org/download/>`_ (choose "macOS 10.13 or later"). Add `/Applications/CMake.app/Contents/bin` to path (for default install). 
  * `Python 3.6 - 3.9 <https://www.python.org/downloads/mac-osx/>`_ (choose 3.6 - 3.9). Install and add to path.
     * Note that OpenVINO is gradually stopping the support for Python 3.6. Python 3.7 - 3.9 are recommended.
  * Apple Xcode Command Line Tools. In the terminal, run `xcode-select --install` from any directory
  * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)

@endsphinxdirective

## Installing OpenVINO Runtime

### <a name="install-core"></a>Step 1: Install OpenVINO Core Components

1. Select and download the OpenVINO™ archive files from [Intel® Distribution of OpenVINO™ toolkit for macOS download page](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-macos). There are typically two files for you to download: 
   ```sh
   m_openvino_toolkit_<operating system>_<release version>_<package ID>_x86_64.tgz
   m_openvino_toolkit_<operating system>_<release version>_<package ID>_x86_64.tgz.sha256
   ``` 
   where the `.sha256` file is used to verify the success of the download process.

2. Locate the downloaded files in your system. This document assumes the files are in your `Downloads` directory. 

3. Open a command prompt terminal window, and verify the checksum of the `sha256` file by using the following command:
   ```sh
   shasum -c -a 256 <archive name>.tgz.sha256
   ```
   If any error message appears, check your network connections, re-download the correct files, and make sure the download process completes successfully.

4. Extract OpenVINO files from the `.tgz` file:
   ```sh
   tar xf <archive name>.tgz -C <destination_dir>
   ```
   where the `<destination_dir>` is the directory that you extract OpenVINO files to. You're recommended to set it as `/opt/intel/`.
   The standard OpenVINO `INSTALL_DIR` referenced in this document is `/opt/intel/openvino_<version>`.

For simplicity, you can create a symbolic link to the latest installation: `/opt/intel/openvino_2022/`. 

The core components are now installed. Continue to the next section to configure environment.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Set environment variables as follows:

```sh
source <INSTALL_DIR>/setupvars.sh
```

If you have more than one OpenVINO™ version on your machine, you can easily switch its version by sourcing `setupvars.sh` of your choice.

> **NOTE**: You can also run this script every time when you start new terminal session. Open `~/.bashrc` in your favorite editor, and add `source <INSTALL_DIR>/setupvars.sh`. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have many OpenVINO™ versions on your machine and want to switch among them, as each may require different setup.

The environment variables are set. Continue to the next section if you want to download any additional components.

### <a name="model-optimizer"></a>Step 3 (Optional): Install Additional Components

Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter are not part of the installer. The OpenVINO™ Development Tools can only be installed via PyPI now. See [Install OpenVINO™ Development Tools](installing-model-dev-tools.md) for detailed steps. 

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on Github](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

### <a name="configure-ncs2"></a>Step 4 (Optional): Configure the Intel® Neural Compute Stick 2 

@sphinxdirective

If you want to run inference on Intel® Neural Compute Stick 2 use the following instructions to setup the device: :ref:`NCS2 Setup Guide <ncs guide macos>`.

@endsphinxdirective

## <a name="get-started"></a>What's Next?

Now you are ready to try out the toolkit. You can use the following tutorials to write your applications using Python and C++.

Start with some Python tutorials:
   * [Hello Image Classification](https://docs.openvino.ai/latest/notebooks/001-hello-world-with-output.html)
   * [Convert TensorFlow models with OpenVINO™](https://docs.openvino.ai/latest/notebooks/101-tensorflow-to-openvino-with-output.html)
   * [Convert a PyTorch model and remove the image background](https://docs.openvino.ai/latest/notebooks/205-vision-background-removal-with-output.html)

To start with C++ samples, see <a href="openvino_docs_OV_UG_Samples_Overview.html#build-samples-macos">Build Sample Applications on macOS</a> first, and then you can try the following samples:
   * [Hello Classification C++ Sample](@ref openvino_inference_engine_samples_hello_classification_README)
   * [Hello Reshape SSD C++ Sample](@ref openvino_inference_engine_samples_hello_reshape_ssd_README)
   * [Image Classification Async C++ Sample](@ref openvino_inference_engine_samples_classification_sample_async_README)

## <a name="uninstall"></a>Uninstalling the Intel® Distribution of OpenVINO™ Toolkit

To uninstall the toolkit, follow the steps on the [Uninstalling page](uninstalling-openvino.md).

@sphinxdirective
.. raw:: html

   </div>

@endsphinxdirective

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
