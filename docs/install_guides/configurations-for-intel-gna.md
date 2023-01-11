# Configurations for Intel® Gaussian & Neural Accelerator (GNA) with OpenVINO™ {#openvino_docs_install_guides_configurations_for_intel_gna}

@sphinxdirective

..note:: On platforms where Intel® GNA is not enabled in the BIOS, the driver cannot be installed, so the GNA plugin uses the software emulation mode only.

Drivers and Dependencies
========================

Intel® GNA hardware requires a driver to be installed on the system.

.. _gna guide:

Linux
=====

Prerequisites
-------------

Ensure that make, gcc, and Linux kernel headers are installed.

Configuration steps
-------------------

#. Download `Intel® GNA driver for Ubuntu Linux 18.04.3 LTS (with HWE Kernel version 5.4+) <https://storage.openvinotoolkit.org/drivers/gna/>`__
#. Run the sample_install.sh script provided in the installation package:

   .. code-block:: sh

   prompt$ ./scripts/sample_install.sh


You can also build and install the driver manually by using the following commands:

.. code-block:: sh

   prompt$ cd src/
   prompt$ make
   prompt$ sudo insmod intel_gna.ko


To unload the driver:

.. code-block:: sh

   prompt$ sudo rmmod intel_gna


.. _gna guide windows:

Windows
=======

Intel® GNA driver for Windows is available through Windows Update.

@endsphinxdirective

