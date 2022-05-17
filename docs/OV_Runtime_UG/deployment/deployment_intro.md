# Deploy with OpenVINO™ {#openvino_deployment_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_install_guides_deployment_manager_tool
   openvino_docs_deploy_local_distribution

@endsphinxdirective

Once the [OpenVINO application development](../integrate_with_your_application.md) has been finished, application developers usually need to deploy their applications to end users. There are several ways how to achieve that:

- Set a dependency on existing prebuilt packages (so called "centralized distribution"):
    - using Debian / RPM packages - a recommended way for distributions of Linux operating system;
    - using pip package manager on PyPi - a default approach for Python-based applications;
    - using Docker images. If the application should be deployed as a Docker image, use a pre-built OpenVINO™ runtime Docker image as a base image in the Dockerfile for the application container image. For more information about OpenVINO Docker images, refer to the installation guides for [Linux](../../install_guides/installing-openvino-docker-linux.md) and [Windows](../../install_guides/installing-openvino-docker-windows.md). 
Furthermore, to customize OpenVINO Docker image, use [Docker CI Framework](https://github.com/openvinotoolkit/docker_ci) to generate a Dockerfile and built it. 
- Grab a necessary functionality of OpenVINO together with your application (so-called "local distribution"):
    - using [OpenVINO Deployment manager](deployment-manager-tool.md) - providing a convenient way create a distribution package;
    - using advanced [Local distribution](local-distribution.md) approach;
    - using [static version of OpenVINO Runtime linked into the final app](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries).

The table below shows which distribution type can be used, depending on a target operating system:

| Distribution type | Operating systems |
|------- ---------- | ----------------- |
| Debian packages | Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit |
| RMP packages | Red Hat Enterprise Linux 8, 64-bit |
| Docker images | Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit; Red Hat Enterprise Linux 8, 64-bit; Windows Server Core base LTSC 2019, 64-bit; Windows 10, version 20H2, 64-bit |
| PyPi (pip package manager) | See [https://pypi.org/project/openvino/](https://pypi.org/project/openvino/) |
| [OpenVINO Deployment Manager](deployment-manager-tool.md) | All operating systems |
| [Local distribution](local-distribution.md) | All operating systems |
| [Build OpenVINO statically and link into the final app](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) | All operating systems |

Depending on the distribution type, the granularity of OpenVINO packages may vary. For example, PyPi distribution of OpenVINO has a [single 'openvino' package](https://pypi.org/project/openvino/) that contains all the runtime libraries and plugins, while a [Local distribution](local-distribution.md) is a more configurable type, thus providing higher granularity. Below are important details of the set of libraries included in the OpenVINO Runtime package:

![deployment_simplified]

- The main library `openvino` is used by users' C++ applications to link against with. The library provides all OpenVINO Runtime public APIs: OpenVINO API 2.0, both Inference Engine and nGraph APIs (being combined as of API 2.0). For C language applications `openvino_c` is additionally required for distribution.
- The "optional" plugin libraries like `openvino_intel_cpu_plugin` (matching `openvino_.+_plugin` pattern) are used to provide inference capabilities on specific devices or additional capabilities like [Hetero execution](../hetero_execution.md) or [Multi-Device execution](../multi_device.md).
- The "optional" plugin libraries like `openvino_ir_frontnend` (matching `openvino_.+_frontend`) are used to provide capabilities to read models of different file formats like OpenVINO IR, ONNX or PaddlePaddle.

The "optional" means that if the application does not use the capability enabled by the plugin, the plugin library or a package with the plugin is not needed in the final distribution.

The information above covers granularity aspects of most distribution types. More detailed information is required and provided in the [Local Distribution](local-distribution.md).

> **NOTE**: Depending on target OpenVINO devices, provide proper configuration of deployed machines, using the [Configurations for GPU](../../install_guides/configurations-for-intel-gpu.md), the [Configurations for GNA](../../install_guides/configurations-for-intel-gna.md), the [Configurations for NCS2](../../install_guides/configurations-for-ncs2.md) or the [Configurations for VPU](../../install_guides/installing-openvino-config-ivad-vpu.md).

[deployment_simplified]: ../../img/deployment_simplified.png
