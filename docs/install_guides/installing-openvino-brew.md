# Install OpenVINO™ Runtime via Homebrew {#openvino_docs_install_guides_installing_openvino_brew}

With the OpenVINO™ 2022.3 release, you can install OpenVINO Runtime on macOS and Linux via [Homebrew](https://brew.sh/). See the [Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino-2022-3-lts-relnotes.html) for more information on updates in the latest release.

Installing OpenVINO Runtime from Homebrew is recommended for C++ developers. If you are working with Python, the PyPI package has everything needed for Python development and deployment on CPU and GPUs. Visit the [Install OpenVINO from PyPI](installing-openvino-pip.md) page for instructions on how to install OpenVINO Runtime for Python using PyPI.

> **NOTE**: Only CPU is supported for inference if you install OpenVINO via HomeBrew.

> **NOTE**: Since the OpenVINO™ 2022.1 release, the following development tools: Model Optimizer, Post-Training Optimization Tool, Model Downloader and other Open Model Zoo tools, Accuracy Checker, and Annotation Converter can be installed via [pypi.org](https://pypi.org/project/openvino-dev/) only.

## Prerequisites

### System Requirements

@sphinxdirective

Full requirement listing is available on the `System Requirements Page <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`_

@endsphinxdirective

### Software Requirements

@sphinxdirective

.. tab:: macOS

  * `Homebrew <https://brew.sh/>`_
  * `CMake 3.13 or higher <https://cmake.org/download/>`_ (choose "macOS 10.13 or later"). Add `/Applications/CMake.app/Contents/bin` to path (for default installation). 
  * `Python 3.7 - 3.10 <https://www.python.org/downloads/mac-osx/>`_ (choose 3.7 - 3.10). Install and add it to path.
  * Apple Xcode Command Line Tools. In the terminal, run `xcode-select --install` from any directory to install it.
  * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)

.. tab:: Linux

  * `Homebrew <https://brew.sh/>`_
  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`_

@endsphinxdirective

## Installing OpenVINO Runtime

@sphinxdirective

1. Make sure that you have installed HomeBrew on your system. If not, follow the instructions on `the Homebrew website <https://brew.sh/>`_ to install and configure it.

2. Open a command prompt terminal window, and run the following command to install OpenVINO Runtime:

   .. code-block:: sh

      brew install openvino


Congratulations, you've finished the installation!

@endsphinxdirective


## (Optional) Installing Additional Components

OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you install OpenVINO Runtime using archive files, OpenVINO Development Tools must be installed separately.

See the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page for step-by-step installation instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the [instructions on GitHub](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

## Uninstalling OpenVINO

To uninstall OpenVINO via HomeBrew, use the following command:
```sh
brew uninstall openvino
```

## What's Next?

@sphinxdirective

Now that you've installed OpenVINO Runtime, you can try the following things: 

* Learn more about :doc:`OpenVINO Workflow <openvino_workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <openvino_docs_model_processing_introduction>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Learn more about :doc:`Inference with OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.

@endsphinxdirective

## Additional Resources

@sphinxdirective

* Intel® Distribution of OpenVINO™ toolkit home page: https://software.intel.com/en-us/openvino-toolkit.
* For IoT Libraries & Code Samples, see the `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`_.

@endsphinxdirective
