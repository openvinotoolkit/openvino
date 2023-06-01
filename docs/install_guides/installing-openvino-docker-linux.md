# Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

@sphinxdirective

Supported Operating Systems for Docker Base Image:

- Ubuntu 22.04 LTS
- Ubuntu 20.04 LTS
- RedHat UBI 8

Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs. You can easily get started by using the precompiled and published docker images. The details on how to get started can be find `here <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__.

In order to start using them you need to meet the following prerequisites:

- Linux operating system or Windows Subsystem for Linux (WSL2)
- Installed docker engine or compatible container engine
- Permissions to start containers (sudo or docker group membership)

As `Docker <https://docs.docker.com/>`__ is (mostly) just an isolation tool, the OpenVINO toolkit inside the container is the same as the OpenVINO toolkit installed natively on the host machine,
so the `OpenVINO documentation <https://docs.openvino.ai/>`__ is fully applicable to containerized OpenVINO distribution.

.. note:: 

   OpenVINO development environment in a docker container is available also in `notebook repository <https://github.com/openvinotoolkit/openvino_notebooks>`__ . It can be deployed in `OpenShift RedHat OpenData Science (RHODS) <https://github.com/openvinotoolkit/operator/blob/main/docs/notebook_in_rhods.md>`__.

You can find more details about Docker CI framework for Intel® Distribution of OpenVINO™ toolkit `here <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__

* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective


