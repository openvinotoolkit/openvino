Configurations for Intel® NPU with OpenVINO™
===============================================

.. meta::
   :description: Learn how to provide additional configuration for Intel®
                 NPU to work with the OpenVINO™ toolkit on your system.


The Intel® NPU device requires a proper driver to be installed in the system.
Make sure you use the most recent supported driver for your hardware setup.


.. tab-set::

   .. tab-item:: Linux

      The driver is maintained as open source and may be found in the following repository,
      together with comprehensive information on installation and system requirements:
      `github.com/intel/linux-npu-driver <https://github.com/intel/linux-npu-driver>`__

      It is recommended to check for the latest version of the driver.

      Make sure you use a supported OS version, as well as install make, gcc,
      and Linux kernel headers. To check the NPU state, use the ``dmesg``
      command in the console. A successful boot-up of the NPU should give you
      a message like this one:

      ``[  797.193201] [drm] Initialized intel_vpu 0.<version number> for 0000:00:0b.0 on minor 0``

      The current requirement for inference on NPU is the minimum of Ubuntu 22.04, kernel
      version of 6.6.

   .. tab-item:: Windows

      The Intel® NPU driver for Windows is available through Windows Update but
      it may also be installed manually by downloading the
      `NPU driver package <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__ and following the
      `Windows driver installation guide <https://support.microsoft.com/en-us/windows/update-drivers-manually-in-windows-ec62f46c-ff14-c91d-eead-d7126dc1f7b6>`__.

      If a driver has already been installed you should be able to find
      'Intel(R) NPU Accelerator' in Windows Device Manager. If you
      cannot find such a device, the NPU is most likely listed in "Other devices"
      as "Multimedia Video Controller."
