---
sidebar_position: 5
sidebar_label: "Install Intel® Distribution of OpenVINO™ Toolkit From a Docker Image"
description: "Learn how to use a prebuilt Docker image or create an image manually to install OpenVINO™ Runtime on Linux and Windows operating systems."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Install Intel® Distribution of OpenVINO™ Toolkit From a Docker Image

This guide presents information on how to use a pre-built Docker image or create a new image
manually, to install OpenVINO™ Runtime.

You can get started easily with pre-built and published docker images, which are available at:

- [Docker Hub](https://hub.docker.com/u/openvino)
- [Red Hat Quay.io](https://quay.io/organization/openvino)
- [Red Hat Ecosystem Catalog (runtime image)](https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3)
- [Red Hat Ecosystem Catalog (development image)](https://catalog.redhat.com/software/containers/intel/openvino-dev/613a450dc9bc35f21dc4a1f7)

:::note
The Ubuntu20 and Ubuntu22 Docker images (runtime and development) now include the tokenizers
and GenAI CPP modules. The development versions of these images also have the Python modules
for these components pre-installed.
:::

You can use the [available Dockerfiles on GitHub](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles)
or generate a Dockerfile with your settings via [DockerHub CI framework](https://github.com/openvinotoolkit/docker_ci/),
which can generate a Dockerfile, build, test, and deploy an image using the Intel® Distribution of OpenVINO™ toolkit.

You can reuse available Dockerfiles, add your layer and customize the OpenVINO™ image to your needs.
The Docker CI repository includes the following guides:

- [Get started with docker images](https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md)
- How to use OpenVINO™ Toolkit containers with [GPU accelerators](https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md) and [NPU accelerators](https://github.com/openvinotoolkit/docker_ci/blob/master/docs/npu_accelerator.md).

To start using Dockerfiles, install Docker Engine or a compatible container
engine on your system:

<Tabs>
  <TabItem value="linux" label="Linux">
  - [Docker Desktop](https://docs.docker.com/desktop/install/linux/)
  - [Docker Engine](https://docs.docker.com/engine/install/)
  </TabItem>
  <TabItem value="win" label="Windows (WSL2)">
  OpenVINO can be installed under [Windows Subsystem for Linux (WSL2)](configurations/configurations-intel-gpu.md#wsl-install).

  - [Docker Desktop](https://docs.docker.com/desktop/install/linux/)
  </TabItem>
</Tabs>

Also, verify you have permissions to run containers (sudo or docker group membership).

:::note
OpenVINO's [Docker](https://docs.docker.com/) and [Bare Metal](index.md)
distributions are identical, so the documentation applies to both.

Note that Ubuntu docker images are no longer provided, Debian-based ones are available instead.
:::

:::note
OpenVINO development environment in a docker container is also available in the
[notebook repository](https://github.com/openvinotoolkit/openvino_notebooks).
It can be implemented in
[OpenShift RedHat OpenData Science (RHODS)](https://github.com/openvinotoolkit/operator/blob/main/docs/notebook_in_rhods.md).
:::

More information about Docker CI for Intel® Distribution of OpenVINO™ toolset can be found
[here](https://github.com/openvinotoolkit/docker_ci/blob/master/README.md)

- [Docker CI framework for Intel® Distribution of OpenVINO™ toolkit](https://github.com/openvinotoolkit/docker_ci/blob/master/README.md)
- [Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit](https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md)
- [Using OpenVINO™ Toolkit containers with GPU accelerators](https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md)
- [Dockerfiles with Intel® Distribution of OpenVINO™ toolkit](https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md)
