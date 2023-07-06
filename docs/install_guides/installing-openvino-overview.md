# Install Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_overview}

@sphinxdirective

.. meta::
   :description: You can choose to install OpenVINO Runtime package - a core set 
                 of libraries or OpenVINO Development Tools - a set of utilities 
                 for working with OpenVINO.


.. toctree::
   :maxdepth: 3
   :hidden:

   OpenVINO Runtime <openvino_docs_install_guides_install_runtime>
   OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>
   Build from Source <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>
   Creating a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>


.. raw:: html

   <script type="module" crossorigin src="_static/selector-tool/assets/index-f34d1fad.js"></script>
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <iframe id="selector" src="_static/selector-tool/selector-136759b.html" style="width: 100%; border: none" title="Download Intel® Distribution of OpenVINO™ Toolkit"></iframe>


Distribution channels of OpenVINO may differ slightly, with regard to supported hardware or available APIs (read installation guides for particular distributions for more details). 
Moreover, OpenVINO Runtime and OpenVINO Development Tools offer different sets of tools, as follows:

* **OpenVINO Runtime** contains the core set of libraries for running machine learning model inference on processor devices.
* **OpenVINO Development Tools** is a set of utilities for working with OpenVINO and OpenVINO models. It includes the following tools:
  - Model Optimizer
  - Post-Training Optimization Tool
  - Benchmark Tool
  - Accuracy Checker and Annotation Converter
  - Model Downloader and other Open Model Zoo tools


Install OpenVINO Development Tools (recommended)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The best way to get started with OpenVINO is to install OpenVINO Development Tools, which will also install the OpenVINO Runtime Python package as a dependency. Follow the instructions on the :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>` page to install it.

**Python**

For developers working in Python, OpenVINO Development Tools can easily be installed using PyPI. See the :ref:`For Python Developers <python_developers>` section of the Install OpenVINO Development Tools page for instructions.

**C++**

For developers working in C++, the core OpenVINO Runtime libraries must be installed separately. Then, OpenVINO Development Tools can be installed using requirements files or PyPI. See the :ref:`For C++ Developers <cpp_developers>` section of the Install OpenVINO Development Tools page for instructions.

Install OpenVINO Runtime only
+++++++++++++++++++++++++++++++++++++++

OpenVINO Runtime may also be installed on its own without OpenVINO Development Tools. This is recommended for users who already have an optimized model and want to deploy it in an application that uses OpenVINO for inference on their device. To install OpenVINO Runtime only, follow the instructions on the :doc:`Install OpenVINO Runtime <openvino_docs_install_guides_install_runtime>` page.

The following methods are available to install OpenVINO Runtime:

* Linux: You can install OpenVINO Runtime using APT, YUM, archive files or Docker. See :doc:`Install OpenVINO on Linux <openvino_docs_install_guides_installing_openvino_linux_header>`.
* Windows: You can install OpenVINO Runtime using archive files or Docker. See :doc:`Install OpenVINO on Windows <openvino_docs_install_guides_installing_openvino_windows_header>`.
* macOS: You can install OpenVINO Runtime using archive files or Docker. See :doc:`Install OpenVINO on macOS <openvino_docs_install_guides_installing_openvino_macos_header>`.

Build OpenVINO from source
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

