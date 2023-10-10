# Install Intel® Distribution of OpenVINO™ toolkit from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker}

@sphinxdirective

.. meta::
   :description: Learn how to use a prebuilt Docker image or create an image 
                 manually to install OpenVINO™ Runtime on Linux and Windows operating systems.

This guide presents information on how to use a pre-built Docker image/create an image manually to install OpenVINO™ Runtime.

Supported host operating systems for the Docker Base image: 

- Linux
- Windows (WSL2)
- macOS (CPU exectuion only)

You can get started easily with pre-built and published docker images, which are available at:

* `Docker Hub <https://hub.docker.com/u/openvino>`__
* `Red Hat Quay.io <https://quay.io/organization/openvino>`__
* `Red Hat Ecosystem Catalog (runtime image) <https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3>`__
* `Red Hat Ecosystem Catalog (development image) <https://catalog.redhat.com/software/containers/intel/openvino-dev/613a450dc9bc35f21dc4a1f7>`__
* `Azure Marketplace <https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvino>`__

You can use the `available Dockerfiles on GitHub <https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles>`__
or generate a Dockerfile with your settings via `DockerHub CI framework <https://github.com/openvinotoolkit/docker_ci/>`__,
which can generate a Dockerfile, build, test, and deploy an image using the Intel® Distribution of OpenVINO™ toolkit. 
You can reuse available Dockerfiles, add your layer and customize the OpenVINO™ image to your needs. 
The Docker CI repository includes guides on how to 
`get started with docker images <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__ and how to use 
`OpenVINO™ Toolkit containers with GPU accelerators. <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md>`__

To start using Dockerfiles the following conditions must be met:

- Linux OS or Windows (under :ref:`Windows Subsystem for Linux (WSL2) <wsl-install>`)
- Installed docker engine or compatible container engine
- Permissions to run containers (sudo or docker group membership)

.. note::

   OpenVINO's `Docker <https://docs.docker.com/>`__ and :doc:`Bare Metal <openvino_docs_install_guides_overview>` 
   distributions are identical, so the documentation applies to both. 

.. note:: 

   OpenVINO development environment in a docker container is also available in the 
   `notebook repository <https://github.com/openvinotoolkit/openvino_notebooks>`__.    It can be implemented in
   `OpenShift RedHat OpenData Science (RHODS) <https://github.com/openvinotoolkit/operator/blob/main/docs/notebook_in_rhods.md>`__.

More information about Docker CI for Intel® Distribution of OpenVINO™ toolset can be found 
`here <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__

* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Using OpenVINO™ Toolkit containers with GPU accelerators <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective


