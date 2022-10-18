# Install OpenVINO™ Runtime on macOS from an Archive File {#openvino_docs_install_guides_installing_openvino_from_archive_macos}

With the OpenVINO™ 2022.2 release, you can download and use archive files to install OpenVINO Runtime. The archive files contain pre-built binaries and library files needed for OpenVINO Runtime, as well as sample code for running demos. This page provides instructions showing how to install OpenVINO Runtime using archive files. Check the [Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes) for more information on updates in the 2022.2 release.

Installing OpenVINO Runtime from archive files is recommended for C++ developers. If you are working with Python, the PyPI package has everything needed for Python development and deployment on CPU and GPUs. Visit the [Install OpenVINO from PyPI](installing-openvino-pip.md) page for instructions on how to install OpenVINO Runtime for Python using PyPI.

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
  * `Python 3.6 - 3.10 <https://www.python.org/downloads/mac-osx/>`_ (choose 3.6 - 3.10). Install and add to path.
     * Note that OpenVINO is gradually phasing out the support for Python 3.6. Python 3.7 - 3.10 are recommended.
  * Apple Xcode Command Line Tools. In the terminal, run `xcode-select --install` from any directory
  * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)

@endsphinxdirective

## Installing OpenVINO Runtime

### <a name="install-core"></a>Step 1: Install OpenVINO Core Components

First, open a terminal. Create a folder for OpenVINO and move into it by issuing the following commands. If the `/opt/intel` folder already exists, skip the `mkdir` command.

```sh
sudo mkdir /opt/intel
cd /opt/intel
```

> **NOTE**: The `/opt/intel` path is the recommended folder path for installing OpenVINO. You may use a different path if desired.

Next, you'll download the OpenVINO Runtime 2022.2 archive file from the [OpenVINO archives](https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/windows) site. Issue the following commands to download the archive file, unpack it, and rename the folder to `openvino_2022.2.0.7713`:

```sh
sudo curl -L https://github.com/openvinotoolkit/openvino/releases/download/2022.2.0/m_openvino_toolkit_osx_2022.2.0.7713.af16ea1d79a_x86_64.tgz --output openvino_2022.2.0.7713.tgz
sudo tar -xf openvino_2022.2.0.7713.tgz
sudo mv m_openvino_toolkit_osx_2022.2.0.7713.af16ea1d79a_x86_64 openvino_2022.2.0.7713
```

Finally, create a symbolic link to the folder by issuing:

```
sudo ln -s openvino_2022.2.0.7713 openvino_2022
```
> **NOTE**: If you have already installed a previous release of OpenVINO 2022, a symbolic link to the `openvino_2022` folder may already exist. Remove the previous link with `sudo rm openvino_2022`, then re-issue the previous command.

Congratulations, you finished installation! The `/opt/intel/openvino_2022` folder now contains the core components for OpenVINO™. When other pages in OpenVINO™ documentation refer to the `<INSTALL_DIR>` directory, this is the folder they're referring to. If you installed OpenVINO™ in a different location, make sure to use that instead.

### <a name="set-the-environment-variables"></a>Step 2: Configure the Environment

You must update several environment variables before you can compile and run OpenVINO™ applications. Open a terminal (if it isn't already open) and run the setupvars.sh script as shown below to temporarily set your environment variables. Again, if you installed OpenVINO™ in a folder other than `/opt/intel/openvino_2022`, use that location instead.

```sh
source /opt/intel/openvino_2022/setupvars.sh
```  

If you have more than one OpenVINO™ version on your machine, you can easily switch its version by sourcing `setupvars.sh` of your choice.

> **NOTE**: The above command must be re-run every time the terminal is opened. To set up Linux so it automatically runs the command every time a new terminal is opened, open `~/.bashrc` in your favorite editor and add `source /opt/intel/openvino_2022/setupvars.sh` after the last line. Next time when you open a terminal, you will see `[setupvars.sh] OpenVINO™ environment initialized`. Changing `.bashrc` is not recommended when you have multiple OpenVINO™ versions on your machine and want to switch among them.

The environment variables are set. Continue to the next section if you want to download any additional components.

### <a name="model-optimizer"></a>Step 3 (Optional): Install Additional Components

OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Runtime using archive files, OpenVINO Development Tools must be installed separately.

See the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page for step-by-step installation instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on Github](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

### <a name="configure-ncs2"></a>Step 4 (Optional): Configure the Intel® Neural Compute Stick 2 

@sphinxdirective

If you want to run inference on Intel® Neural Compute Stick 2 use the following instructions to setup the device: :ref:`NCS2 Setup Guide <ncs guide macos>`.

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
