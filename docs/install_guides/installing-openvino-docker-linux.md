# Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image {#openvino_docs_install_guides_installing_openvino_docker_linux}


@sphinxdirective

This guide provides steps on creating a Docker image with Intel® Distribution of OpenVINO™ toolkit for Linux and using the image on different devices. 

System Requirements
###################

.. tab:: Target Operating Systems with Python Versions
  
   +----------------------------------------------+-------------------------+
   | Operating System                             | Included Python Version |
   +==============================================+=========================+
   | Ubuntu 18.04 long-term support (LTS), 64-bit |  3.8                    |
   +----------------------------------------------+-------------------------+
   | Ubuntu 20.04 long-term support (LTS), 64-bit |  3.8                    |
   +----------------------------------------------+-------------------------+
   | Red Hat Enterprise Linux 8, 64-bit           |  3.8                    |
   +----------------------------------------------+-------------------------+

.. tab:: Host Operating Systems

   * Linux
   * Windows Subsystem for Linux 2 (WSL2) on CPU or GPU
   * macOS on CPU only
   
   To launch a Linux image on WSL2 when trying to run inferences on a GPU, make sure that the following requirements are met:
 
   * Only Windows 10 with 21H2 update or above installed and Windows 11 are supported.
   * Intel GPU driver for Windows, version 30.0.100.9684 or newer needs to be installed. For more details, refer to
     `this article at intel.com <https://www.intel.com/content/www/us/en/artificial-intelligence/harness-the-power-of-intel-igpu-on-your-machine.html#articleparagraph_983312434>`__.
   * Currently, the Docker images contain preinstalled recommended version of OpenCL Runtime with WSL2 support.


Installation
#############

* Use a prebuilt image:
  
  1. `Get a prebuilt image from provided sources <#getting-a-prebuilt-image-from-provided-sources>`__
  2. `Run the image on different devices <#running-the-docker-image-on-different-devices>`__
  3. `Run samples in the Docker image <#running-samples-in-docker-image>`__

* If you want to customize your image, you can also build a Docker image manually:
  
  1. `Prepare a Dockerfile <#preparing-a-dockerfile>`__
  2. `Configure the Docker image <#configuring-the-image-for-different-devices>`__
  3. `Run the image on different devices <#running-the-docker-image-on-different-devices>`__
  4. `Run samples in the Docker image <#running-samples-in-docker-image>`__


Getting a Prebuilt Image from Provided Sources
++++++++++++++++++++++++++++++++++++++++++++++

You can find prebuilt images on:

- `Docker Hub <https://hub.docker.com/u/openvino>`__
- `Red Hat Quay.io <https://quay.io/organization/openvino>`__
- `Red Hat Ecosystem Catalog (runtime image) <https://catalog.redhat.com/software/containers/intel/openvino-runtime/606ff4d7ecb5241699188fb3>`__
- `Red Hat Ecosystem Catalog (development image) <https://catalog.redhat.com/software/containers/intel/openvino-dev/613a450dc9bc35f21dc4a1f7>`__
- `Azure Marketplace <https://azuremarketplace.microsoft.com/en-us/marketplace/apps/intel_corporation.openvino>`__

Preparing a Dockerfile
++++++++++++++++++++++

You can use the `available Dockerfiles on GitHub <https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles>`__
or generate a Dockerfile with your settings via `DockerHub CI Framework <https://github.com/openvinotoolkit/docker_ci>`__
which can generate a Dockerfile, build, test and deploy an image with the Intel® Distribution of OpenVINO™ toolkit.
You can also try our `Tutorials <https://github.com/openvinotoolkit/docker_ci/tree/master/docs/tutorials>`__ 
which demonstrate the usage of Docker containers with OpenVINO. 

Configuring the Image for Different Devices
+++++++++++++++++++++++++++++++++++++++++++

If you want to run inference on a CPU no extra configuration is needed. 
Go to `Run the image on different devices <running-the-docker-image-on-different-devices>`__ for the next step.

If you want to run inference on a GPU, follow the instructions provided in the guide on 
:doc:`Configuration for Intel GPU <openvino_docs_install_guides_configurations_for_intel_gpu>`


Running the Docker Image on Different Devices
+++++++++++++++++++++++++++++++++++++++++++++

Running the Image on CPU
-------------------------

Run the Docker image with the following command:

.. code-block:: sh

   docker run -it --rm <image_name>


Note the following:

- Kernel reports the same information for all containers as for native application, 
  for example, CPU, memory information.
- All instructions that are available to host process available for process in container, 
  including, for example, AVX2, AVX512. No restrictions.
- Docker does not use virtualization or emulation. The process in Docker is just a regular 
  Linux process, but it is isolated from external world on kernel level. Performance loss is minor.


Running the Image on GPU
-------------------------

.. note:: 
  
   Only Intel® integrated graphics are supported.

Note the following:

- GPU is not available in the container by default. You must attach it to the container.
- Kernel driver must be installed on the host.
- In the container, non-root user must be in the ``video`` and ``render`` groups. 
  To add a user to the render group, follow the 
  `Configuration Guide for the Intel® Graphics Compute Runtime for OpenCL™ on Ubuntu 20.04 <https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md>`__.

To make GPU available in the container, attach the GPU to the container using ``--device /dev/dri`` option and run the container:

* Ubuntu 18 or RHEL 8:
  
  .. code-block:: sh

     docker run -it --rm --device /dev/dri <image_name>

  .. note:: 
   
     If your host system is Ubuntu 20, follow the 
     `Configuration Guide for the Intel® Graphics Compute Runtime for OpenCL™ on Ubuntu* 20.04 <https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md>`__.

* WSL2:
  
  .. code-block:: sh

     docker run -it --rm --device /dev/dxg --volume /usr/lib/wsl:/usr/lib/wsl <image_name>

  .. note::
   
     To launch a Linux image on WSL2, make sure that the additional `System Requirements <system-requirements>`__ are met.


Running Samples in Docker Image
###############################

To run the ``Hello Classification Sample`` on a specific inference device, run the following commands:

**CPU**:

.. code-block:: sh

   docker run -it --rm <image_name>
   /bin/bash -c "cd ~ && omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 /opt/intel/openvino/samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp CPU"

**GPU**:

.. code-block:: sh

   docker run -itu root:root  --rm --device /dev/dri:/dev/dri <image_name>
   /bin/bash -c "omz_downloader --name googlenet-v1 --precisions FP16 && omz_converter --name googlenet-v1 --precision FP16 && curl -O https://storage.openvinotoolkit.org/data/test_data/images/car_1.bmp && python3 samples/python/hello_classification/hello_classification.py public/googlenet-v1/FP16/googlenet-v1.xml car_1.bmp GPU"


Additional Resources
###############################

- `DockerHub CI Framework <https://github.com/openvinotoolkit/docker_ci>`__ for Intel® Distribution of OpenVINO™ toolkit. 
  The Framework can generate a Dockerfile, build, test, and deploy an image with the Intel® Distribution of OpenVINO™ toolkit. 
  You can reuse available Dockerfiles, add your layer and customize the image of OpenVINO™ for your needs.
- `Intel® Distribution of OpenVINO™ toolkit home page <https://software.intel.com/en-us/openvino-toolkit>`__
- `OpenVINO Installation Selector Tool <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html>`__


@endsphinxdirective


