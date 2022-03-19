# Deployment with OpenVINO {#documentation}

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_install_guides_deployment_manager_tool
   openvino_docs_deploy_local_distribution

@endsphinxdirective

Once the [OpenVINO application development](../integrate_with_your_application.md) has finished, usually users need to deploy their applications to end users. There are several ways how to achieve that:

- Centralized distribution, when all applications will reuse the same OpenVINO package in bounds of the same OpenVINO version (e.i. if several applications require several different OpenVINO versions, several packages are installed; but if several applications work on top of the version of OpenVINO runtime - a single packages is installed):
    - Using Debian / RPM packages is a recommended way for a family of Linux operation systems
    - Using pip package manager on PyPi, default approach for Python-based applications
- Local distribution, when all applications will use their own copies of OpenVINO libraries independently on OpenVINO version:
    - Using Docker images
    - Using [OpenVINO Deployment manager](deployment-manager-tool.md) providing a convinient way for local distribution
    - Using advanced [Local distribution](local-distribution.md), the approach works for all operation systems

The table below shows which distribution type can be used depending on target operation system:

| Distribution type | Operation systems |
|------- ---------- | ----------------- |
| Debian packages | Ubuntu 18.04 long-term support (LTS), 64-bit; Ubuntu 20.04 long-term support (LTS), 64-bit |
| RMP packages | Red Hat Enterprise Linux 8, 64-bit |
| Docker images | All operation systems |
| PyPi (pip package manager) | All operation systems |
| [OpenVINO Deployment Manager](deployment-manager-tool.md) | All operation systems |
| [Local distribution](local-distribution.md) | All operation systems |

Dependning on the distribution type, the granularity of OpenVINO packages may vary: PyPi distribution [OpenVINO has a single package `openvino`](https://pypi.org/project/openvino/) containing all the runtime libraries and plugins, while more configurable ways like [Local distribution](local-distribution.md) provide higher granularity, so it is important to now some details about the set of libraries which are part of OpenVINO Runtime package:

- The main library `openvino` is used by C++ user's applications to link against with. The library provides all OpenVINO Runtime public API for both OpenVINO API 2.0 and Inference Engine, nGraph APIs.
    > **NOTE**: for C language applications `openvino_c` is additionally required for distribution

- The libraries like `openvino_intel_cpu_plugin` (matching `openvino_.+_plugin` pattern) are used to provide inference capabilities on specific devices or additional capabitilies like [Hetero execution](../hetero_execution.md) or [Multi-Device execution](../multi_device.md).
- The libraries like `openvino_ir_frontnend` (matching `openvino_.+_frontend`) are used to provide capabilities to read models of different file formats like OpenVINO IR, ONNX or Paddle.

This information covers granularity aspects of majority distribution types, more information is only needed and provided in [Local Distribution](local-distribution.md).

> **NOTE**: Depending on target OpenVINO devices, you also have to use [Configurations for GPU](../../install_guides/configurations-for-intel-gpu.md), [Configurations for GNA](../../install_guides/configurations-for-intel-gna.md), [Configurations for NCS2](../../install_guides/configurations-for-ncs2.md) or [Configurations for GNA](../../install_guides/installing-openvino-config-ivad-vpu.md) for proper configuration of deployed machines.
