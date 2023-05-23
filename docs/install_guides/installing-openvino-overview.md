# Installing Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_overview}

@sphinxdirective

.. toctree::
   :maxdepth: 3
   :hidden:
   
   OpenVINO Runtime <openvino_docs_install_guides_install_runtime>
   OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>
   Build from Source <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>
   Creating a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>

Intel® Distribution of OpenVINO™ toolkit is a comprehensive toolkit for developing applications and solutions based on deep learning tasks, such as computer vision, automatic speech recognition, natural language processing, recommendation systems, and more. It provides high-performance and rich deployment options, from edge to cloud. Some of its advantages are:

* Enables CNN-based and transformer-based deep learning inference on the edge or cloud.
* Supports various execution modes across Intel® technologies: Intel® CPU, Intel® Integrated Graphics, Intel® Discrete Graphics, and more.
* Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels.
* Compatible with models from a wide variety of frameworks, including TensorFlow, PyTorch, PaddlePaddle, ONNX, and more.


Install OpenVINO
################

.. button-link:: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
   :color: primary
   :outline:

   Check out the OpenVINO Download Page :fas:`fa-external-link-alt`


OpenVINO installation package is distributed in two parts: OpenVINO Runtime and OpenVINO Development Tools.

* **OpenVINO Runtime** contains the core set of libraries for running machine learning model inference on processor devices.
* **OpenVINO Development Tools** is a set of utilities for working with OpenVINO and OpenVINO models. It includes the following tools:
  - Model Optimizer
  - Post-Training Optimization Tool
  - Benchmark Tool
  - Accuracy Checker and Annotation Converter
  - Model Downloader and other Open Model Zoo tools

Option 1. Install OpenVINO Runtime and OpenVINO Development Tools (recommended)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The best way to get started with OpenVINO is to install OpenVINO Development Tools, which will also install the OpenVINO Runtime Python package as a dependency. Follow the instructions on the :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>` page to install it.

**Python**

For developers working in Python, OpenVINO Development Tools can easily be installed using PyPI. See the :ref:`For Python Developers <openvino_docs_install_guides_install_dev_tools.html#python_developers>` section of the Install OpenVINO Development Tools page for instructions.

**C++**

For developers working in C++, the core OpenVINO Runtime libraries must be installed separately. Then, OpenVINO Development Tools can be installed using requirements files or PyPI. See the :ref:`For C++ Developers <openvino_docs_install_guides_install_dev_tools.html#cpp_developers>` section of the Install OpenVINO Development Tools page for instructions.

Option 2. Install OpenVINO Runtime only
+++++++++++++++++++++++++++++++++++++++

OpenVINO Runtime may also be installed on its own without OpenVINO Development Tools. This is recommended for users who already have an optimized model and want to deploy it in an application that uses OpenVINO for inference on their device. To install OpenVINO Runtime only, follow the instructions on the :doc:`Install OpenVINO Runtime <openvino_docs_install_guides_install_runtime>` page.

The following methods are available to install OpenVINO Runtime:

* Linux: You can install OpenVINO Runtime using APT, YUM, archive files or Docker. See :doc:`Install OpenVINO on Linux <openvino_docs_install_guides_installing_openvino_linux_header>`.
* Windows: You can install OpenVINO Runtime using archive files or Docker. See :doc:`Install OpenVINO on Windows <openvino_docs_install_guides_installing_openvino_windows_header>`.
* macOS: You can install OpenVINO Runtime using archive files or Docker. See :doc:`Install OpenVINO on macOS <openvino_docs_install_guides_installing_openvino_macos_header>`.
* :doc:`Raspbian OS <openvino_docs_install_guides_installing_openvino_raspbian>`

Option 3. Build OpenVINO from source
++++++++++++++++++++++++++++++++++++

Source files are also available in the OpenVINO Toolkit GitHub repository. If you want to build OpenVINO from source for your platform, follow the `OpenVINO Build Instructions <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__ .

Next Steps
##########

Still unsure if you want to install OpenVINO toolkit? Check out the :doc:`OpenVINO tutorials <tutorials>` to run example applications directly in your web browser without installing it locally. Here are some exciting demos you can explore:

- `Monodepth Estimation with OpenVINO <notebooks/201-vision-monodepth-with-output.html>`__
- `Live Style Transfer with OpenVINO <notebooks/404-style-transfer-with-output.html>`__
- `OpenVINO API Tutorial <notebooks/002-openvino-api-with-output.html>`__

Follow these links to install OpenVINO:

- :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>`
- :doc:`Install OpenVINO Runtime <openvino_docs_install_guides_install_runtime>`
- `Build from Source <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__

@endsphinxdirective

