.. {#system_requirements}

System Requirements
===================


.. note::

   Certain hardware (including but not limited to GPU, GNA, and latest CPUs) requires manual
   installation of specific drivers and/or other software components to work correctly and/or
   to utilize hardware capabilities at their best. This might require updates to operating
   system, including but not limited to Linux kernel, please refer to their documentation
   for details. These modifications should be handled by user and are not part of OpenVINO
   installation.


CPU
##########

.. tab-set::

   .. tab-item:: Supported Hardware

      * Intel Atom® processor with Intel® SSE4.2 support
      * Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
      * 6th - 14th generation Intel® Core™ processors
      * Intel® Core™ Ultra (codename Meteor Lake)
      * 1st - 5th generation Intel® Xeon® Scalable Processors
      * ARM and ARM64 CPUs; Apple M1, M2, and Raspberry Pi

   .. tab-item:: Supported Operating Systems

      * Ubuntu 22.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 20.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 18.04 long-term support (LTS) with limitations, 64-bit (Kernel 5.4+)
      * Windows 10
      * Windows 11
      * macOS 12.6 and above, 64-bit and ARM64
      * CentOS 7
      * Red Hat Enterprise Linux 8, 64-bit
      * Ubuntu 18 ARM64
      * Debian 9 ARM

GPU
##########

.. tab-set::

   .. tab-item::  Supported Hardware

      * Intel® HD Graphics
      * Intel® UHD Graphics
      * Intel® Iris® Pro Graphics
      * Intel® Iris® Xe Graphics
      * Intel® Iris® Xe Max Graphics
      * Intel® Arc™ GPU Series
      * Intel® Data Center GPU Flex Series
      * Intel® Data Center GPU Max Series

   .. tab-item::  Supported Operating Systems

      * Ubuntu 22.04 long-term support (LTS), 64-bit
      * Ubuntu 20.04 long-term support (LTS), 64-bit
      * Windows 10, 64-bit
      * Windows 11, 64-bit
      * CentOS 7
      * Red Hat Enterprise Linux 8, 64-bit

   .. tab-item:: Additional considerations

      * The use of of GPU requires drivers that are not included in the Intel®
        Distribution of OpenVINO™ toolkit package.
      * A chipset that supports processor graphics is required for Intel® Xeon®
        processors. Processor graphics are not included in all processors. See
        `Product Specifications  <https://ark.intel.com/>`__
        for information about your processor.
      * While this release of OpenVINO supports Ubuntu 20.04, the driver stack
        for Intel discrete graphic cards does not fully support Ubuntu 20.04.
        We recommend using Ubuntu 22.04 when executing on discrete graphics.
      * The following minimum (i.e., used for old hardware) OpenCL™ driver's versions
        were used during OpenVINO internal validation: 22.43 for Ubuntu 22.04, 21.48
        for Ubuntu 20.04 and 21.49 for Red Hat Enterprise Linux 8.

Intel® Neural Processing Unit
################################

.. tab-set::

   .. tab-item:: Operating Systems for NPU

      * Ubuntu 22.04 long-term support (LTS), 64-bit
      * Windows 11, 64-bit

   .. tab-item:: Additional considerations

      * These Accelerators require :doc:`drivers <openvino_docs_install_guides_configurations_for_intel_npu>`
        that are not included in the Intel® Distribution of OpenVINO™ toolkit package.
      * Users can access the NPU plugin through the OpenVINO archives on
        the :doc:`download page <openvino_docs_install_guides_overview>`.


Intel® Gaussian & Neural Accelerator
##########################################

.. tab-set::

   .. tab-item:: Operating Systems for GNA

      * Ubuntu 22.04 long-term support (LTS), 64-bit
      * Ubuntu 20.04 long-term support (LTS), 64-bit
      * Windows 10, 64-bit
      * Windows 11, 64-bit


Operating systems and developer environment
#######################################################

.. tab-set::

   .. tab-item:: Linux OS

      * Ubuntu 22.04 with Linux kernel 5.15+
      * Ubuntu 20.04 with Linux kernel 5.15+
      * Red Hat Enterprise Linux 8 with Linux kernel 5.4

      Build environment components:

      * Python* 3.8-3.11
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__
        required for inference on GPU
      * GNU Compiler Collection and CMake are needed for building from source:

        * `GNU Compiler Collection (GCC) <https://www.gnu.org/software/gcc/>`__ 7.5 and above
        * `CMake <https://cmake.org/download/>`__ 3.10 or higher

      Higher versions of kernel might be required for 10th Gen Intel® Core™ Processors, 11th Gen
      Intel® Core™ Processors, 11th Gen Intel® Core™ Processors S-Series Processors, 12th Gen
      Intel® Core™ Processors, 13th Gen Intel® Core™ Processors, 14th Gen
      Intel® Core™ Processors, Intel® Core™ Ultra Processors, 4th Gen Intel® Xeon® Scalable Processors
      or 5th Gen Intel® Xeon® Scalable Processors to support CPU, GPU, GNA or hybrid-cores CPU capabilities.

   .. tab-item:: Windows 10 and 11

      Build environment components:

      * `Microsoft Visual Studio 2019 <https://visualstudio.microsoft.com/vs/older-downloads/>`__
      * `CMake <https://cmake.org/download/>`__ 3.10 or higher
      * `Python* 3.8-3.11 <http://www.python.org/downloads/>`__
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__
        required for inference on GPU

   .. tab-item:: macOS

      * macOS 12.6 and above

      Build environment components:

      * `Xcode* 10.3 <https://developer.apple.com/xcode/>`__
      * `Python* 3.8-3.11 <http://www.python.org/downloads/>`__
      * `CMake <https://cmake.org/download/>`__ 3.10 or higher

   .. tab-item:: DL frameworks versions:

      * TensorFlow 1.15, 2.12
      * ONNX 1.14.1
      * PaddlePaddle 2.4

      This package can be installed on other versions of DL Frameworks
      but only the versions specified here are fully validated.


.. note::

   OpenVINO Python binaries and binaries on Windows, CentOS 7, and macOS (x86) are built
   with oneTBB libraries, and others on Ubuntu and RedHat systems are built with
   legacy TBB which is released by OS distribution. OpenVINO can be built from source
   with either oneTBB or legacy TBB on all the systems listed here. System
   compatibility and performance are improved on Hybrid CPUs
   such as 12th Gen Intel Core and above.


Legal Information
+++++++++++++++++++++++++++++++++++++++++++++

You may not use or facilitate the use of this document in connection with any infringement
or other legal analysis concerning Intel products described herein.

You agree to grant Intel a non-exclusive, royalty-free license to any patent claim
thereafter drafted which includes subject matter disclosed herein.

No license (express or implied, by estoppel or otherwise) to any intellectual property
rights is granted by this document.

All information provided here is subject to change without notice. Contact your Intel
representative to obtain the latest Intel product specifications and roadmaps.

The products described may contain design defects or errors known as errata which may
cause the product to deviate from published specifications. Current characterized errata
are available on request.

Intel technologies' features and benefits depend on system configuration and may require
enabled hardware, software or service activation. Learn more at
`http://www.intel.com/ <http://www.intel.com/>`__
or from the OEM or retailer.

No computer system can be absolutely secure.

Intel, Atom, Arria, Core, Movidius, Xeon, OpenVINO, and the Intel logo are trademarks
of Intel Corporation in the U.S. and/or other countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos

Other names and brands may be claimed as the property of others.

Copyright © 2023, Intel Corporation. All rights reserved.

For more complete information about compiler optimizations, see our Optimization Notice.

Performance varies by use, configuration and other factors. Learn more at
`www.Intel.com/PerformanceIndex <www.Intel.com/PerformanceIndex>`__.







