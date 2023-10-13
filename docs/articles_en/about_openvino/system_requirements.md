# System Requirements {#system_requirements}

@sphinxdirective


Certain hardware (including but not limited to GPU and GNA) requires manual
installation of specific drivers to work correctly. The drivers may also require
updates to the operating system, including Linux kernel. These updates need to
be handled by the user and are not part of OpenVINO installation. Refer to your
system's documentation for updating instructions.


Intel CPU processors 
#####################

.. tab-set::

   .. tab-item:: Supported Hardware

      * Intel Atom® processor with Intel® SSE4.2 support
      * Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics 
      * 6th - 13th generation Intel® Core™ processors 
      * Intel® Xeon® Scalable Processors (code name Skylake)
      * 2nd Generation Intel® Xeon® Scalable Processors (code name Cascade Lake) 
      * 3rd Generation Intel® Xeon® Scalable Processors (code name Cooper Lake and Ice Lake) 
      * 4th Generation Intel® Xeon® Scalable Processors (code name Sapphire Rapids) 

   .. tab-item:: Required Operating Systems

      * Ubuntu 22.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 20.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 18.04 long-term support (LTS) with limitations, 64-bit (Kernel 5.4+)
      * Windows* 10 
      * Windows* 11 
      * macOS* 10.15 and above, 64-bit 
      * Red Hat Enterprise Linux* 8, 64-bit

Intel® Processor Graphics
###########################################

.. tab-set::

   .. tab-item:: Supported Hardware

      * Intel® HD Graphics 
      * Intel® UHD Graphics
      * Intel® Iris® Pro Graphics
      * Intel® Iris® Xe Graphics
      * Intel® Iris® Xe Max Graphics
      * Intel® Arc ™ GPU Series
      * Intel® Data Center GPU Flex Series

   .. tab-item:: Required Operating Systems

      * Ubuntu* 22.04 long-term support (LTS), 64-bit
      * Ubuntu* 20.04 long-term support (LTS), 64-bit
      * Windows* 10, 64-bit
      * Windows* 11, 64-bit
      * Red Hat Enterprise Linux* 8, 64-bit

.. note::

   | Using a GPU requires installing drivers that are not included in the Intel® Distribution of OpenVINO™ toolkit. 
   | Not all Intel CPUs include the integrated graphics processor. See `Product Specifications <https://ark.intel.com/>`__
     for information about your processor. 
   | Although this release works with Ubuntu 20.04 for discrete graphic cards, the support is limited 
     due to discrete graphics drivers. 
   | Recommended `OpenCL™ driver <https://github.com/intel/compute-runtime>`__ versions:
     22.43 for Ubuntu 22.04, 22.41 for Ubuntu 20.04 and 22.28 for Red Hat Enterprise Linux 8 


Intel® Gaussian & Neural Accelerator
###########################################

Operating Systems:

Ubuntu* 22.04 long-term support (LTS), 64-bit
Ubuntu* 20.04 long-term support (LTS), 64-bit
Windows* 10, 64-bit 
Windows* 11, 64-bit
 

Operating system and developer environment requirements
############################################################

.. tab-set::

   .. tab-item:: Linux OS

      * Ubuntu 22.04 with Linux kernel 5.15+
      * Ubuntu 20.04 with Linux kernel 5.15+
      * RHEL 8 with Linux kernel 5.4

      A Linux OS build environment requires:
      
      * Python* 3.7-3.11
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__ 
        for inference on a GPU.

      GNU Compiler Collection and CMake are needed for building from source:

      * `GNU Compiler Collection (GCC) <https://www.gnu.org/software/gcc/>`__
        8.4 (RHEL 8) 9.3 (Ubuntu 20)
      * `CMake <https://cmake.org/download/>`__ 3.10 or higher

      To support CPU, GPU, GNA, or hybrid-core CPU capabilities, higher versions of kernel 
      might be required for 10th Gen Intel® Core™ Processor, 
      11th Gen Intel® Core™ Processors, 11th Gen Intel® Core™ Processors S-Series Processors, 
      12th Gen Intel® Core™ Processors, 13th Gen Intel® Core™ Processors,  or 4th Gen 
      Intel® Xeon® Scalable Processors.

   .. tab-item:: Windows* 10 and 11

      A Windows OS build environment requires:

      * `Microsoft Visual Studio 2019 <https://visualstudio.microsoft.com/vs/older-downloads/>`__
      * `CMake <https://cmake.org/download/>`__ 3.14 or higher
      * `Python 3.7-3.11 <http://www.python.org/downloads/>`__
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__ for inference on a GPU.

   .. tab-item:: macOS* 10.15 and above

      A macOS build environment requires:

      * `Xcode 10.3 <https://developer.apple.com/xcode/>`__
      * `Python 3.7-3.11 <http://www.python.org/downloads/>`__
      * `CMake 3.13 or higher <https://cmake.org/download/>`__

   .. tab-item:: DL framework versions

      * TensorFlow 1.15, 2.12
      * MxNet 1.9
      * ONNX 1.13
      * PaddlePaddle* 2.4

      Other DL Framework versions may be compatible with the current OpenVINO
      release, but only the versions listed here are fully validated.


.. note::

   OpenVINO Python binaries and binaries on Windows/CentOS7/MACOS(x86) are built
   with oneTBB libraries. Other binaries on Ubuntu and Redhat OSes are built with
   legacy TBB which is released by OS distribution. OpenVINO can be built with 
   either oneTBB or legacy TBB by the user on all OS systems listed. System 
   compatibility and performance are improved on Hybrid CPUs, 
   such as 12th Gen Intel Core and above.



@endsphinxdirective