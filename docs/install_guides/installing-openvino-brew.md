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

.. tab:: Linux

  * `Homebrew <https://brew.sh/>`_
  * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
  * GCC 7.5.0 (for Ubuntu 18.04) or GCC 9.3.0 (for Ubuntu 20.04)
  * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`_

.. tab:: macOS

  * `Homebrew <https://brew.sh/>`_
  * `CMake 3.13 or higher <https://cmake.org/download/>`_ (choose "macOS 10.13 or later"). Add `/Applications/CMake.app/Contents/bin` to path (for default installation). 
  * `Python 3.7 - 3.10 <https://www.python.org/downloads/mac-osx/>`_ (choose 3.7 - 3.10). Install and add it to path.
  * Apple Xcode Command Line Tools. In the terminal, run `xcode-select --install` from any directory to install it.
  * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)

@endsphinxdirective

## Installing OpenVINO Runtime

@sphinxdirective

1. Make sure that you have installed HomeBrew on your system. If not, follow the instructions on `the Homebrew website <https://brew.sh/>`_ to install and configure it.

2. Run the following command to install OpenVINO Runtime:

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

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning applications! Learn more about how to integrate a model in OpenVINO applications by trying out the following tutorials.

* Try the `C++ Quick Start Example <openvino_docs_get_started_get_started_demos.html>`_ for step-by-step instructions on building and running a basic image classification C++ application.

  .. image:: https://user-images.githubusercontent.com/36741649/127170593-86976dc3-e5e4-40be-b0a6-206379cd7df5.jpg
     :width: 400


* Visit the :ref:`Samples <code samples>` page for other C++ example applications to get you started with OpenVINO, such as:
   * `Basic object detection with the Hello Reshape SSD C++ sample <openvino_inference_engine_samples_hello_reshape_ssd_README.html>`_
   * `Automatic speech recognition C++ sample <openvino_inference_engine_samples_speech_sample_README.html>`_

@endsphinxdirective

## Additional Resources

@sphinxdirective

* To convert models for use with OpenVINO, see :doc:`Model Optimizer Developer Guide <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Try out OpenVINO via `OpenVINO Notebooks <https://docs.openvino.ai/2022.3/notebooks/notebooks.html>`_.
* To write your own OpenVINO™ applications, see :doc:`OpenVINO Runtime User Guide <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO™ Toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.
* Intel® Distribution of OpenVINO™ toolkit home page: https://software.intel.com/en-us/openvino-toolkit.
* For IoT Libraries & Code Samples see the `Intel® IoT Developer Kit <https://github.com/intel-iot-devkit>`_.

@endsphinxdirective
