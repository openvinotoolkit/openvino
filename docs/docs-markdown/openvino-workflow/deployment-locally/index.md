---
sidebar_position: 4
sidebar_label: "Deploy Locally"
description: "There are several ways of deploying OpenVINO™ application once its development has been finished."
---

# Deploy Locally


:::note
Note that [running inference in OpenVINO Runtime](../running-inference/index.md) is the most basic form of deployment. Before moving forward, make sure you know how to create a proper Inference configuration and [develop your application properly](../running-inference/index.md).
:::

## Local Deployment Options

- Set a dependency on the existing prebuilt packages, also called "centralized distribution":
  
  - using Debian / RPM packages - a recommended way for Linux operating systems;
  - using PIP package manager on PyPI - the default approach for Python-based applications;
  - using Docker images - if the application should be deployed as a Docker image, use a pre-built OpenVINO™ Runtime Docker image as a base image in the Dockerfile for the application container image. For more information about OpenVINO Docker images, refer to [Installing OpenVINO from Docker](../../get-started/install-openvino/install-openvino-docker-linux.md)
    
    - Furthermore, to customize your OpenVINO Docker image, use the [Docker CI Framework](https://github.com/openvinotoolkit/docker_ci) to generate a Dockerfile and build the image.
- Grab a necessary functionality of OpenVINO together with your application, also called "local distribution":
  
  - using the [local distribution](local-distribution-libraries.md) approach;
  - using [a static version of OpenVINO Runtime linked to the final app](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libraries.md).

The table below shows which distribution type can be used for what target operating system:

| Distribution type | Operating systems |
| --- | --- |
| Debian packages | 20.04, 22.04, 24.04 (64-bit) |
| RPM packages | Red Hat Enterprise Linux 8, 64-bit |
| Docker images | Ubuntu 20.04, 22.04, 24.04 (64-bit); Red Hat Enterprise Linux 8, 64-bit |
| PyPI (PIP package manager) | See [https://pypi.org/project/openvino](https://pypi.org/project/openvino) |
| [Libraries for Local Distribution](local-distribution-libraries.md) | All operating systems |
| [Build OpenVINO statically and link to the final app](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libraries.md) | All operating systems |

## Granularity of Major Distribution Types

The granularity of OpenVINO packages may vary for different distribution types. For example, the PyPI distribution of OpenVINO has a [single 'openvino' package](https://pypi.org/project/openvino/) that contains all the runtime libraries and plugins, while a [local distribution](local-distribution-libraries.md) is a more configurable type providing higher granularity. Below are important details of the set of libraries included in the OpenVINO Runtime package:

![](deployment_simplified.svg)

- The main library `openvino` is used by users' C++ applications to link against with. For C language applications, `openvino_c` is additionally required for distribution. The library includes OpenVINO API 2.0.
- The "optional" plugin libraries like `openvino_intel_cpu_plugin` (matching the `openvino_.+_plugin` pattern) are used to provide inference capabilities on specific devices or additional capabilities like [Hetero Execution](../running-inference/inference-devices-and-modes/hetero-execution.md).
- The "optional" plugin libraries like `openvino_ir_frontend` (matching `openvino_.+_frontend`) are used to provide capabilities to read models of different file formats such as OpenVINO IR, TensorFlow, ONNX, and PaddlePaddle.

Here the term "optional" means that if the application does not use the capability enabled by the plugin, the plugin library or a package with the plugin is not needed in the final distribution.

Building a local distribution will require more detailed information, and you will find it in the dedicated [Libraries for Local Distribution](local-distribution-libraries.md) article.

:::note
Depending on your target OpenVINO devices, the following configuration might be needed for deployed machines: [Configurations for GPU](../../get-started/install-openvino/configurations/configurations-intel-gpu.md).
:::
