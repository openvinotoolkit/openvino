# Install Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_overview}

@sphinxdirective

.. meta::
   :description: You can choose to install OpenVINO Runtime package - a core set 
                 of libraries or OpenVINO Development Tools - a set of utilities 
                 for working with OpenVINO.


.. toctree::
   :maxdepth: 3
   :hidden:

   OpenVINO Runtime on Linux <openvino_docs_install_guides_installing_openvino_linux_header>
   OpenVINO Runtime on Windows <openvino_docs_install_guides_installing_openvino_windows_header>
   OpenVINO Runtime on macOS <openvino_docs_install_guides_installing_openvino_macos_header>  
   OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>
   Create a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>


.. raw:: html

   <script type="module" crossorigin src="_static/selector-tool/assets/index-f34d1fad.js"></script>
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <iframe id="selector" src="_static/selector-tool/selector-136759b.html" style="width: 100%; border: none" title="Download Intel® Distribution of OpenVINO™ Toolkit"></iframe>


Different OpenVINO distributions may differ with regard to supported hardware or available APIs.
Read installation guides for particular distributions for more details. 

| **OpenVINO Runtime:** 
|    contains the core set of libraries for running inference on various processing units. It is recommended for users who already have an optimized model 
     and want to deploy it in an application using OpenVINO for inference on their devices.

| **OpenVINO Development Tools:** 
|    includes the OpenVINO Runtime for Python, as well as a set of utilities for optimizing models and validating performance. 
     It is recommended for users who want to optimize and verify their models before applying them in their applications.
     For Python developers it is ready out-of-the-box, while for C++ development you need to install OpenVINO Runtime libraries separately.
|    See the :ref:`For C++ Developers <cpp_developers>` section of the install guide for detailed instructions.
|    Development Tools provides:
     * Model conversion API
     * Benchmark Tool
     * Accuracy Checker and Annotation Converter
     * Post-Training Optimization Tool
     * Model Downloader and other Open Model Zoo tools


| **Build OpenVINO from source**
|    OpenVINO Toolkit source files are available on GitHub as open source. If you want to build your own version of OpenVINO for your platform, 
     follow the `OpenVINO Build Instructions <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md>`__ .




@endsphinxdirective

