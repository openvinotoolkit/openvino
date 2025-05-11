Install Intel® Distribution of OpenVINO™ Toolkit From a Docker Image
=======================================================================

.. meta::
   :description: Learn how to use a prebuilt Docker image or create an image
                 manually to install OpenVINO™ Runtime on Linux and Windows operating systems.

This guide presents information on how to use a pre-built Docker image or create a new image
manually, to install OpenVINO™ Runtime.

You can get started easily with pre-built and published docker images, which are available at:

* `Docker Hub <https://hub.docker.com/u/openvino>`__
* `Red Hat Quay.io <https://quay.io/organization/openvino>`__
* `Red Hat Ecosystem Catalog (runtime image) <https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3>`__
* `Red Hat Ecosystem Catalog (development image) <https://catalog.redhat.com/software/containers/intel/openvino-dev/613a450dc9bc35f21dc4a1f7>`__

.. note::

   The Ubuntu20 and Ubuntu22 Docker images (runtime and development) now include the tokenizers
   and GenAI CPP modules. The development versions of these images also have the Python modules
   for these components pre-installed.

You can use the `available Dockerfiles on GitHub <https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles>`__
or generate a Dockerfile with your settings via `DockerHub CI framework <https://github.com/openvinotoolkit/docker_ci/>`__,
which can generate a Dockerfile, build, test, and deploy an image using the Intel® Distribution of OpenVINO™ toolkit.
You can reuse available Dockerfiles, add your layer and customize the OpenVINO™ image to your needs.
The Docker CI repository includes guides on how to
`get started with docker images <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__ and how to use
`OpenVINO™ Toolkit containers with GPU accelerators. <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md>`__

To start using Dockerfiles, install Docker Engine or a compatible container
engine on your system:

.. tab-set::

   .. tab-item:: Linux
      :sync: linux

      * `Docker Desktop <https://docs.docker.com/desktop/install/linux/>`__
      * `Docker Engine <https://docs.docker.com/engine/install/>`__

   .. tab-item:: Windows (WSL2)
      :sync: win

      OpenVINO can be installed under :ref:`Windows Subsystem for Linux (WSL2) <wsl_install>`.

      * `Docker Desktop <https://docs.docker.com/desktop/install/linux/>`__

Also, verify you have permissions to run containers (sudo or docker group membership).

.. note::

   OpenVINO's `Docker <https://docs.docker.com/>`__ and :doc:`Bare Metal <../install-openvino>`
   distributions are identical, so the documentation applies to both.

   Note that Ubuntu docker images are no longer provided, Debian-based ones are available instead.

.. note::

   OpenVINO development environment in a docker container is also available in the
   `notebook repository <https://github.com/openvinotoolkit/openvino_notebooks>`__.
   It can be implemented in
   `OpenShift RedHat OpenData Science (RHODS) <https://github.com/openvinotoolkit/operator/blob/main/docs/notebook_in_rhods.md>`__.

More information about Docker CI for Intel® Distribution of OpenVINO™ toolset can be found
`here <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__

* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Using OpenVINO™ Toolkit containers with GPU accelerators <https://github.com/openvinotoolkit/docker_ci/blob/master/docs/accelerators.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

