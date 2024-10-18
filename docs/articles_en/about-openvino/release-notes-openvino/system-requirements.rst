System Requirements
===================


.. note::

   Certain hardware, including but not limited to GPU and NPU, requires manual installation of
   specific drivers and/or other software components to work correctly and/or to utilize
   hardware capabilities at their best. This might require updates to the operating
   system, including but not limited to Linux kernel, please refer to their documentation
   for details. These modifications should be handled by user and are not part of OpenVINO
   installation.


CPU
##########

.. tab-set::

   .. tab-item:: Supported Hardware

      * Intel® Core™ Ultra Series 1 and Series 2 (Windows only)
      * Intel® Xeon® 6 processor (preview)
      * Intel Atom® Processor X Series
      * Intel Atom® processor with Intel® SSE4.2 support
      * Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
      * 6th - 14th generation Intel® Core™ processors
      * 1st - 5th generation Intel® Xeon® Scalable Processors
      * ARM CPUs with armv7a and higher, ARM64 CPUs with arm64-v8a and higher, Apple® Mac with Apple silicon

   .. tab-item:: Supported Operating Systems

      * Windows 11, 64-bit
      * Windows 10, 64-bit
      * Ubuntu 24.04 long-term support (LTS), 64-bit (Kernel 6.8+) (preview support)
      * Ubuntu 22.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 20.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * macOS 12.6 and above, 64-bit and ARM64
      * CentOS 7
      * Red Hat Enterprise Linux 9.3-9.4, 64-bit
      * openSUSE Tumbleweed, 64-bit and ARM64
      * Ubuntu 20.04 ARM64

GPU
##########

.. tab-set::

   .. tab-item::  Supported Hardware

      * Intel® Arc™ GPU Series
      * Intel® HD Graphics
      * Intel® UHD Graphics
      * Intel® Iris® Pro Graphics
      * Intel® Iris® Xe Graphics
      * Intel® Iris® Xe Max Graphics
      * Intel® Data Center GPU Flex Series
      * Intel® Data Center GPU Max Series

   .. tab-item::  Supported Operating Systems

      * Windows 11, 64-bit
      * Windows 10, 64-bit
      * Ubuntu 24.04 long-term support (LTS), 64-bit
      * Ubuntu 22.04 long-term support (LTS), 64-bit
      * Ubuntu 20.04 long-term support (LTS), 64-bit
      * CentOS 7
      * Red Hat Enterprise Linux 9.3-9.4, 64-bit

   .. tab-item:: Additional considerations

      * The use of GPU requires drivers that are not included in the Intel®
        Distribution of OpenVINO™ toolkit package.
      * Processor graphics are not included in all processors. See
        `Product Specifications <https://ark.intel.com/>`__
        for information about your processor.
      * While this release of OpenVINO supports Ubuntu 20.04, the driver stack
        for Intel discrete graphic cards does not fully support Ubuntu 20.04.
        We recommend using Ubuntu 22.04 and later when executing on discrete graphics.
      * OpenCL™ driver versions required may vary, depending on hardware and operating Systems
        used. Consult driver documentation to select the best version for your setup.

Intel® Neural Processing Unit
################################

.. tab-set::

   .. tab-item:: Operating Systems for NPU

      * Ubuntu 24.04 long-term support (LTS), 64-bit (preview support)
      * Ubuntu 22.04 long-term support (LTS), 64-bit
      * Windows 11, 64-bit (22H2 and later)

   .. tab-item:: Additional considerations

      * These Accelerators require :doc:`drivers <../../get-started/configurations/configurations-intel-npu>`
        that are not included in the Intel® Distribution of OpenVINO™ toolkit package.
      * Users can access the NPU plugin through the OpenVINO archives on
        the :doc:`download page <../../get-started/install-openvino>`.



Operating systems and developer environment
#######################################################

.. tab-set::

   .. tab-item:: Linux OS

      * Ubuntu 24.04 with Linux kernel 6.8+
      * Ubuntu 22.04 with Linux kernel 5.15+
      * Ubuntu 20.04 with Linux kernel 5.15+
      * Red Hat Enterprise Linux 9.3-9.4 with Linux kernel 5.4

      Build environment components:

      * Python 3.8-3.12
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__
        required for inference on GPU
      * GNU Compiler Collection and CMake are needed for building from source:

        * `GNU Compiler Collection (GCC) <https://www.gnu.org/software/gcc/>`__ 7.5 and above
        * `CMake <https://cmake.org/download/>`__ 3.13 or higher

      Higher versions of kernel might be required for 10th Gen Intel® Core™ Processors and above,
      Intel® Core™ Ultra Processors, 4th Gen Intel® Xeon® Scalable Processors and above
      to support CPU, GPU, NPU or hybrid-cores CPU capabilities.

   .. tab-item:: Windows 10 and 11

      Build environment components:

      * `Microsoft Visual Studio 2019 <https://visualstudio.microsoft.com/vs/older-downloads/>`__
      * `CMake <https://cmake.org/download/>`__ 3.16 or higher
      * `Python <http://www.python.org/downloads/>`__ 3.8-3.12
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__
        required for inference on GPU

   .. tab-item:: macOS

      * macOS 12.6 and above

      Build environment components:

      * `Xcode <https://developer.apple.com/xcode/>`__ 10.3
      * `CMake <https://cmake.org/download/>`__ 3.13 or higher
      * `Python <http://www.python.org/downloads/>`__ 3.8-3.12

   .. tab-item:: DL framework versions:

      * TensorFlow 1.15.5 - 2.17
      * PyTorch 2.4
      * ONNX 1.16
      * PaddlePaddle 2.6
      * JAX 0.4.31 (via a path of jax2tf with native_serialization=False)

      This package can be installed on other versions of DL Frameworks
      but only the versions specified here are fully validated.


.. note::

   OpenVINO Python binaries are built with and redistribute oneTBB libraries.



The claims stated here may not apply to all use cases and setups. See
:doc:`Legal notices and terms of use <../additional-resources/terms-of-use>` for more information.