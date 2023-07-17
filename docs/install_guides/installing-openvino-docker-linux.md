# Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker_linux}

@sphinxdirective

.. meta::
   :description: Learn how to use a prebuilt Docker image or create an image 
                 manually to install OpenVINO™ Runtime on Linux and Windows operating systems.


Supported operating systems for the Docker Base image: 

- Ubuntu 22.04 LTS
- Ubuntu 20.04 LTS
- RedHat UBI 8
- Windows (WSL2)

.. important::

   While Windows is listed as a supported system, there is no dedicated Docker Image for it. To work with Windows, use Windows Subsystem for Linux (WSL2).

The `Docker CI framework <https://github.com/openvinotoolkit/docker_ci/>`__ can generate a Dockerfile, build, test, and deploy an image using the Intel® Distribution of OpenVINO™ toolkit. You can reuse available Dockerfiles, add your layer and customize the OpenVINO™ image to your needs. You can get started easily with pre-built and published docker images. Details on how to get started can be found `here <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__.

To start using them, the following conditions must be met:

- Linux OS or Windows (under :ref:`Windows Subsystem for Linux (WSL2) <wsl-install>`)
- Installed docker engine or compatible container engine
- Permissions to run containers (sudo or docker group membership)

OpenVINO's `Docker <https://docs.docker.com/>`__ and `Bare Metal <https://docs.openvino.ai/2023.0/ovms_docs_deploying_server.html#doxid-ovms-docs-deploying-server>`__ distributions are identical, so the documentation applies to both.

.. note:: 

   The OpenVINO development environment in a docker container is also available in the `notebook repository <https://github.com/openvinotoolkit/openvino_notebooks>`__ . It can be implemented in `OpenShift RedHat OpenData Science (RHODS) <https://github.com/openvinotoolkit/operator/blob/main/docs/notebook_in_rhods.md>`__.

More information about Docker CI for Intel® Distribution of OpenVINO™ toolset can be found `here <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__

* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Using OpenVINO™ Toolkit containers with GPU accelerators <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective


