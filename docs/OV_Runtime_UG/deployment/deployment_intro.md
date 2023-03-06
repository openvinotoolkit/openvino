# Run and Deploy Locally {#openvino_deployment_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Run Inference <openvino_docs_OV_UG_OV_Runtime_User_Guide>
   Optimize Inference <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>
   Deploy Application with Deployment Manager <openvino_docs_install_guides_deployment_manager_tool>
   Local Distribution Libraries <openvino_docs_deploy_local_distribution>

@endsphinxdirective

> **NOTE**: Note that [running inference in OpenVINO Runtime](../openvino_intro.md) is the most basic form of deployment. Before moving forward, make sure you know how to create a proper Inference configuration and [develop your application properly](../integrate_with_your_application.md)


## Local Deployment Options

- Set a dependency on the existing prebuilt packages, also called "centralized distribution":
    - using Debian / RPM packages - a recommended way for Linux operating systems;
    - using PIP package manager on PyPI - the default approach for Python-based applications;
    - using Docker images - if the application should be deployed as a Docker image, use a pre-built OpenVINO™ Runtime Docker image as a base image in the Dockerfile for the application container image. For more information about OpenVINO Docker images, refer to [Installing OpenVINO on Linux from Docker](../../install_guides/installing-openvino-docker-linux.md) and [Installing OpenVINO on Windows from Docker](../../install_guides/installing-openvino-docker-windows.md). 
Furthermore, to customize your OpenVINO Docker image, use the [Docker CI Framework](https://github.com/openvinotoolkit/docker_ci) to generate a Dockerfile and built the image. 
- Grab a necessary functionality of OpenVINO together with your application, also called "local distribution":
    - using [OpenVINO Deployment Manager](deployment-manager-tool.md) - providing a convenient way for creating a distribution package;
    - using the advanced [local distribution](local-distribution.md) approach;
    - using [a static version of OpenVINO Runtime linked to the final app](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries).

The table below shows which distribution type can be used for what target operating system:

| Distribution type | Operating systems |
|------- ---------- | ----------------- |
| Debian packages | Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit |
| RMP packages | Red Hat Enterprise Linux 8, 64-bit |
| Docker images | Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit; Red Hat Enterprise Linux 8, 64-bit; Windows Server Core base LTSC 2019, 64-bit; Windows 10, version 20H2, 64-bit |
| PyPI (PIP package manager) | See [https://pypi.org/project/openvino/](https://pypi.org/project/openvino/) |
| [OpenVINO Deployment Manager](deployment-manager-tool.md) | All operating systems |
| [Local distribution](local-distribution.md) | All operating systems |
| [Build OpenVINO statically and link to the final app](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) | All operating systems |

## Granularity of Major Distribution Types

The granularity of OpenVINO packages may vary for different distribution types. For example, the PyPI distribution of OpenVINO has a [single 'openvino' package](https://pypi.org/project/openvino/) that contains all the runtime libraries and plugins, while a [local distribution](local-distribution.md) is a more configurable type providing higher granularity. Below are important details of the set of libraries included in the OpenVINO Runtime package:

![](../../img/deployment_simplified.svg)

- The main library `openvino` is used by users' C++ applications to link against with. The library provides all OpenVINO Runtime public APIs, including both API 2.0 and the previous Inference Engine and nGraph APIs. For C language applications, `openvino_c` is additionally required for distribution.
- The "optional" plugin libraries like `openvino_intel_cpu_plugin` (matching the `openvino_.+_plugin` pattern) are used to provide inference capabilities on specific devices or additional capabilities like [Hetero Execution](../hetero_execution.md) and [Multi-Device Execution](../multi_device.md).
- The "optional" plugin libraries like `openvino_ir_frontend` (matching `openvino_.+_frontend`) are used to provide capabilities to read models of different file formats such as OpenVINO IR, TensorFlow, ONNX, and PaddlePaddle.

Here the term "optional" means that if the application does not use the capability enabled by the plugin, the plugin library or a package with the plugin is not needed in the final distribution.

Building a local distribution will require more detailed information, and you will find it in the dedicated [Libraries for Local Distribution](local-distribution.md) article.

> **NOTE**: Depending on your target OpenVINO devices, the following configurations might be needed for deployed machines: [Configurations for GPU](../../install_guides/configurations-for-intel-gpu.md), [Configurations for GNA](../../install_guides/configurations-for-intel-gna.md).
