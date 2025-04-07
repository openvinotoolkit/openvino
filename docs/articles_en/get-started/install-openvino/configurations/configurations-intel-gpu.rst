Configurations for Intel® Processor Graphics (GPU) with OpenVINO™
====================================================================


.. meta::
   :description: Learn how to provide additional configuration for Intel®
                 Processor Graphics (GPU) to work with Intel® Distribution of
                 OpenVINO™ toolkit on your system.


.. _gpu guide:

To use the OpenVINO™ GPU plug-in and transfer the inference to the graphics of the Intel® processor (GPU), the Intel® graphics driver must be properly configured on the system.

Linux
#####

To use a GPU device for OpenVINO inference, you must install OpenCL runtime packages.

If you are using a discrete GPU (for example Arc 770), you must also be using a supported Linux kernel as per `documentation. <https://dgpu-docs.intel.com/driver/kernel-driver-types.html>`__

- For Arc GPU, kernel 6.2 or higher is recommended.
- For Max and Flex GPU, or Arc with kernel version lower than 6.2, you must also install the ``intel-i915-dkms`` and ``xpu-smi`` kernel modules as described in the installation documentation for `Max/Flex <https://dgpu-docs.intel.com/driver/installation.html>`__ or `Arc. <https://dgpu-docs.intel.com/driver/client/overview.html>`__

Below are the instructions on how to install the OpenCL packages on supported Linux distributions. These instructions install the `Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver <https://github.com/intel/compute-runtime/releases/tag/23.22.26516.18>`__ and its dependencies:

- `Intel Graphics Memory Management Library <https://github.com/intel/gmmlib>`__
- `Intel® Graphics Compiler for OpenCL™ <https://github.com/intel/intel-graphics-compiler>`__
- `OpenCL ICD loader package <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`__

.. tab-set::

   .. tab-item:: Ubuntu 22.04 LTS / Ubuntu 24.04 LTS
      :sync: ubuntu-22

      Download and install the `deb` packages published `here <https://github.com/intel/compute-runtime/releases/latest>`__
      and install the apt package `ocl-icd-libopencl1` with the OpenCl ICD loader.

      Alternatively, you can add the apt repository by following the
      `installation guide <https://dgpu-docs.intel.com/driver/installation.html#ubuntu>`__.
      Then install the `ocl-icd-libopencl1`, `intel-opencl-icd`, `intel-level-zero-gpu` and `level-zero`
      apt packages:

      .. code-block:: sh

         apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
         sudo usermod -a -G render $LOGNAME

   .. tab-item:: Ubuntu 20.04 LTS
      :sync: ubuntu-20

      Ubuntu 20.04 LTS is not updated with the latest driver versions. You can install the updated versions up to the version 22.43 from apt:

      .. code-block:: sh

         apt-get update && apt-get install -y --no-install-recommends curl gpg gpg-agent && \
         curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
         echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
         apt-get update
         apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero
         sudo usermod -a -G render $LOGNAME

      Alternatively, download older `deb` version from `here <https://github.com/intel/compute-runtime/releases>`__. Note that older driver version might not include some of the bug fixes and might be not supported on some latest platforms. Check the supported hardware for the versions you are installing.

   .. tab-item:: RedHat UBI 8
      :sync: redhat-8

      Follow the `guide <https://dgpu-docs.intel.com/driver/installation.html#rhel-install-steps>`__ to add Yum repository.

      Install following packages:

      .. code-block:: sh

         yum install intel-opencl level-zero intel-level-zero-gpu intel-igc-core intel-igc-cm intel-gmmlib intel-ocloc

      Install the OpenCL ICD Loader via:

      .. code-block:: sh

         rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm

.. _gpu guide windows:

Windows
#######

To install the Intel Graphics Driver for Windows, follow the `driver installation instructions <https://www.intel.com/content/www/us/en/support/articles/000005629/graphics.html>`_.

To check if the driver has been installed:

1. Type **device manager** in the **Search Windows** field and press Enter. **Device Manager** will open.
2. Click the drop-down arrow to display **Display Adapters**. You can see the adapter that is installed in your computer:

   .. image:: ../../../assets/images/DeviceManager.PNG
      :width: 400

3. Right-click on the adapter name and select **Properties**.
4. Click the **Driver** tab to view the driver version.

   .. image:: ../../../assets/images/DeviceDriverVersion.svg
      :width: 400

Your device driver has been updated and is now ready to use your GPU.

.. _wsl_install:

Windows Subsystem for Linux (WSL)
#################################

WSL allows developers to run a GNU/Linux development environment for the Windows operating system. Using the GPU in WSL is very similar to a native Linux environment.

.. note::

   Make sure your Intel graphics driver is updated to version **30.0.100.9955** or later. You can download and install the latest GPU host driver `here <https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html>`__.

Below are the required steps to make it work with OpenVINO:

- Install the GPU drivers as described :ref:`above <wsl_install>`.
- Run the following commands in PowerShell to view the latest version of WSL2:

  .. code-block:: sh

     wsl --update
     wsl --shutdown

- When booting Ubuntu 20.04, 22.04, or 24.04 install the same drivers as described above in the Linux section

.. note::

   In WSL, the GPU device is accessed via the character device `/dev/drx`, while for native Linux OS it is accessed via `/dev/dri`.

Additional Resources
####################

* :doc:`GPU Device <../../../openvino-workflow/running-inference/inference-devices-and-modes/gpu-device>`
* :doc:`Install Intel® Distribution of OpenVINO™ toolkit from a Docker Image <../install-openvino-archive-linux>`
* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__
* `GPU Driver issue troubleshoot <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_plugin_driver_troubleshooting.md>`


