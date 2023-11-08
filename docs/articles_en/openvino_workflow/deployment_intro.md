# Deploy Locally {#openvino_deployment_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Deploy Application with Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>
   Local Distribution Libraries <openvino_docs_deploy_local_distribution>
   Optimize Binaries Size <opevino_conditional_compilation_deploying_guide>

.. meta::
   :description: There are several ways of deploying OpenVINO™ application once 
                 its development has been finished.


.. note::

   Note that :doc:`running inference in OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>` is the most basic form of deployment. Before moving forward, make sure you know how to create a proper Inference configuration and :doc:`develop your application properly <openvino_docs_OV_UG_Integrate_OV_with_your_application>`.

Local Deployment Options
########################

- Set a dependency on the existing prebuilt packages, also called "centralized distribution":

  - using Debian / RPM packages - a recommended way for Linux operating systems;
  - using PIP package manager on PyPI - the default approach for Python-based applications;
  - using Docker images - if the application should be deployed as a Docker image, use a pre-built OpenVINO™ Runtime Docker image as a base image in the Dockerfile for the application container image. For more information about OpenVINO Docker images, refer to :doc:`Installing OpenVINO from Docker <openvino_docs_install_guides_installing_openvino_docker>`
- Furthermore, to customize your OpenVINO Docker image, use the `Docker CI Framework <https://github.com/openvinotoolkit/docker_ci>`__ to generate a Dockerfile and built the image.
- Grab a necessary functionality of OpenVINO together with your application, also called "local distribution":

  - using :doc:`OpenVINO Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>` - providing a convenient way for creating a distribution package;
  - using the advanced :doc:`local distribution <openvino_docs_deploy_local_distribution>` approach;
  - using `a static version of OpenVINO Runtime linked to the final app <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libaries.md>`__.

The table below shows which distribution type can be used for what target operating system:

.. list-table::
   :header-rows: 1

   * - Distribution type
     - Operating systems
   * - Debian packages
     - Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit
   * - RPM packages
     - Red Hat Enterprise Linux 8, 64-bit
   * - Docker images
     - Ubuntu 22.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit; Red Hat Enterprise Linux 8, 64-bit
   * - PyPI (PIP package manager)
     - See https://pypi.org/project/openvino
   * - :doc:`OpenVINO Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>`
     - All operating systems
   * - :doc:`Libraries for Local Distribution <openvino_docs_deploy_local_distribution>`
     - All operating systems
   * - `Build OpenVINO statically and link to the final app <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libaries.md>`__
     - All operating systems


Granularity of Major Distribution Types
#######################################

The granularity of OpenVINO packages may vary for different distribution types. For example, the PyPI distribution of OpenVINO has a `single 'openvino' package <https://pypi.org/project/openvino/>`__ that contains all the runtime libraries and plugins, while a :doc:`local distribution <openvino_docs_deploy_local_distribution>` is a more configurable type providing higher granularity. Below are important details of the set of libraries included in the OpenVINO Runtime package:

.. image:: _static/images/deployment_simplified.svg


- The main library ``openvino`` is used by users' C++ applications to link against with. The library provides all OpenVINO Runtime public APIs, including both API 2.0 and the previous Inference Engine and nGraph APIs. For C language applications, ``openvino_c`` is additionally required for distribution.
- The "optional" plugin libraries like ``openvino_intel_cpu_plugin`` (matching the ``openvino_.+_plugin`` pattern) are used to provide inference capabilities on specific devices or additional capabilities like :doc:`Hetero Execution <openvino_docs_OV_UG_Hetero_execution>` and :doc:`Multi-Device Execution <openvino_docs_OV_UG_Running_on_multiple_devices>`.
- The "optional" plugin libraries like ``openvino_ir_frontend`` (matching ``openvino_.+_frontend``) are used to provide capabilities to read models of different file formats such as OpenVINO IR, TensorFlow, ONNX, and PaddlePaddle.

Here the term "optional" means that if the application does not use the capability enabled by the plugin, the plugin library or a package with the plugin is not needed in the final distribution.

Building a local distribution will require more detailed information, and you will find it in the dedicated :doc:`Libraries for Local Distribution <openvino_docs_deploy_local_distribution>` article.

.. note::

   Depending on your target OpenVINO devices, the following configurations might be needed for deployed machines: :doc:`Configurations for GPU <openvino_docs_install_guides_configurations_for_intel_gpu>`, :doc:`Configurations for GNA <openvino_docs_install_guides_configurations_for_intel_gna>`.

@endsphinxdirective