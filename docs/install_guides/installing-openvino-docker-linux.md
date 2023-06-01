# Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

@sphinxdirective

Supported operating systems for the Docker Base image:

- Ubuntu 22.04 LTS
- Ubuntu 20.04 LTS
- RedHat UBI 8

The framework can generate a Dockerfile, build, test, and deploy an image using the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the OpenVINO™ image to your needs. You can get started easily with pre-built and published docker images. Details on how to get started can be found `here <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__.

To start using them, the following conditions must be met:

- Linux OS or Windows Subsystem for Linux (WSL2)
- Installed docker engine or compatible container engine
- Permissions to run containers (sudo or docker group membership)

Since `Docker <https://docs.docker.com/>`__ is (mostly) just an isolation tool, the OpenVINO toolkit in the container is the same as the OpenVINO toolkit installed natively on the host machine, so the `OpenVINO documentation <https://docs.openvino.ai/>`__ documentation is fully applicable to the containerized OpenVINO distribution.

.. note:: 

   The OpenVINO development environment in a docker container is also available in the `notebook repository <https://github.com/openvinotoolkit/openvino_notebooks>`__ . It can be implemented in `OpenShift RedHat OpenData Science (RHODS) <https://github.com/openvinotoolkit/operator/blob/main/docs/notebook_in_rhods.md>`__.

ore information about Docker CI for Intel® Distribution of OpenVINO™ toolset can be found `here <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__

* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective


