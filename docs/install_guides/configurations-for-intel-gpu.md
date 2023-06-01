# Configurations for Intel® Processor Graphics (GPU) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gpu}

@sphinxdirective

.. _gpu guide:

To use the OpenVINO™ GPU plugin and offload inference to Intel® Processor Graphics (GPU), Intel® Graphics Driver must be properly configured on your system.

Linux
#####

To use GPU device for inference with OpenVINO you need to meet the following prerequisites:

- use supported Linux kernel, according to the `documentation <https://dgpu-docs.intel.com/driver/kernel-driver-types.html>`__
- install GPU runtime drivers: `The Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver <https://github.com/intel/compute-runtime/releases/latest>`__
- install `Intel Graphics Memory Management Library <https://github.com/intel/gmmlib>`__ and `Intel® Graphics Compiler for OpenCL™ <https://github.com/intel/intel-graphics-compiler>`__
- install `OpenCl ICD loader package <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`__

Depending on OS, there might be different methods for installing the packages above. Below are instructions on how to install the packages, divided by OS on which You want to install.

Ubuntu 22.04 LTS
++++++++++++++++

Download and install the `deb` packages published `here <https://github.com/intel/compute-runtime/releases/latest>`__ and install apt package `ocl-icd-libopencl1` with OpenCl ICD loader.

Alternatively, add apt repository (described `here <https://dgpu-docs.intel.com/driver/installation.html>`__). For example, for ARC device use command:

.. code-block:: sh
   
   apt-get update && apt-get install -y gpg gpg-agent curl
   curl https://repositories.intel.com/graphics/intel-graphics.key | \
     gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
   echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | \
     tee  /etc/apt/sources.list.d/intel.gpu.jammy.list


Install the `ocl-icd-libopencl1`, `intel-opencl-icd`, `intel-level-zero-gpu` and `level-zero` apt packages:

.. code-block:: sh

   apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero

Ubuntu 20.04 LTS
++++++++++++++++

Ubuntu 20.04 LTS is not updated with the latest driver versions. You can install the updated versions up to the version 22.43 from apt:

.. code-block:: sh
   
   apt-get update && apt-get install -y --no-install-recommends curl gpg gpg-agent && \
   curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
   echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
   apt-get update
   apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero

Alternatively, download older `deb` version from `here <https://github.com/intel/compute-runtime/releases>`__. Note that older driver version might not include some of the bug fixes and might be not supported on some latest platforms. Check the supported hardware for the versions you are installing.

RedHat UBI 8
++++++++++++

Follow the `instructions <https://dgpu-docs.intel.com/driver/installation.html#red-hat-enterprise-linux-8-6>`__ to add Yum repository.

Install following packages: 

* `intel-opencl`,
* `level-zero`,
* `intel-level-zero-gpu` ,
* `intel-igc-core`, 
* `intel-igc-cm`, 
* `intel-gmmlib`,
* `intel-ocloc`.

Install the OpenCL ICD Loader via:

.. code-block:: sh

   rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm

.. _gpu guide windows:

Windows
#######

To install the Intel Graphics Driver for Windows, follow the `driver installation guide <https://www.intel.com/content/www/us/en/support/articles/000005629/graphics.html>`_.

To check if driver has been installed:

1. Type **device manager** in your **Search Windows** box and press Enter. The **Device Manager** opens.
2. Click the drop-down arrow to view the **Display adapters**. You can see the adapter that is installed in your computer:  

   .. image:: _static/images/DeviceManager.PNG
      :width: 400

3. Right-click the adapter name and select **Properties**.
4. Click the **Driver** tab to see the driver version. 

   .. image:: _static/images/DeviceDriverVersion.svg
      :width: 400

Your device driver has been updated and it is ready to use your GPU now.

Windows Subsystem for Linux (WSL)
#################################

WSL allows developers to run a GNU/Linux development environment for Windows OS. Using GPU in WSL is very similar to the native Linux environment.

.. note::

   Make sure that Intel Graphics Driver is updated to **30.0.100.9955** or later version. You can download and install the latest GPU host driver `here <https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html>`__.

Below are the required steps to make it work with OpenVINO:

- Install GPU drivers as described above.
- Run the following commands in PowerShell to se the latest version of WSL2:

  .. code-block:: sh

     wsl --update
     wsl --shutdown
  
- While Ubuntu 20.04 or Ubuntu 22.04 is started, install the same drivers like described above in the Linux section

.. note:: 
   
   In WSL, GPU device is accessible via a character device `/dev/drx`, while for a native Linux OS vie `/dev/dri`.

Additional Resources
####################

Following driver versions of Intel® Graphics Driver were used in the OpenVINO internal validation:

+------------------+-------------------------------------------------------------------------------------------+
| Operation System | Driver version                                                                            |
+==================+===========================================================================================+
| Ubuntu 22.04     | `22.43.24595.30 <https://github.com/intel/compute-runtime/releases/tag/22.43.24595.30>`__ |
+------------------+-------------------------------------------------------------------------------------------+
| Ubuntu 20.04     | `22.35.24055 <https://github.com/intel/compute-runtime/releases/tag/22.35.24055>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| Ubuntu 18.04     | `21.38.21026 <https://github.com/intel/compute-runtime/releases/tag/21.38.21026>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| CentOS 7         | `19.41.14441 <https://github.com/intel/compute-runtime/releases/tag/19.41.14441>`__       |
+------------------+-------------------------------------------------------------------------------------------+
| RHEL 8           | `22.28.23726 <https://github.com/intel/compute-runtime/releases/tag/22.28.23726>`__       |
+------------------+-------------------------------------------------------------------------------------------+


What’s Next?
############

* :doc:`GPU Device <openvino_docs_OV_UG_supported_plugins_GPU>`
* :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image <openvino_docs_install_guides_installing_openvino_docker_linux>`
* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective


