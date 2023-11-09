# System Requirements {#system_requirements}

@sphinxdirective


Certain hardware requires specific drivers to work properly with OpenVINO. 
These drivers, including Linux* kernels, might require updates to your operating system,
which is not part of OpenVINO installation. Refer to your hardware's documentation 
for updating instructions.


CPU
##########

.. tab-set::

   .. tab-item:: Supported Processors

      * Intel Atom® processor with Intel® SSE4.2 support
      * Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
      * 6th - 13th generation Intel® Core™ processors
      * Intel® Core™ Ultra (codename Meteor Lake)
      * Intel® Xeon® Scalable Processors (code name Skylake) 
      * 2nd Generation Intel® Xeon® Scalable Processors (code name Cascade Lake)
      * 3rd Generation Intel® Xeon® Scalable Processors (code name Cooper Lake and Ice Lake)
      * 4th Generation Intel® Xeon® Scalable Processors (code name Sapphire Rapids)
      * ARM and ARM64 CPUs 

   .. tab-item:: Operating Systems for CPU

      * Ubuntu 22.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 20.04 long-term support (LTS), 64-bit (Kernel 5.15+)
      * Ubuntu 18.04 long-term support (LTS) with limitations, 64-bit (Kernel 5.4+)
      * Windows* 10
      * Windows* 11
      * macOS* 10.15 and above, 64-bit
      * macOS 11 and above, ARM64
      * Red Hat Enterprise Linux* 8, 64-bit
      * Debian 9 ARM64 and ARM
      * CentOS 7 64-bit 

GPU
##########

.. tab-set::

   .. tab-item:: Supported Processors

      * Intel® HD Graphics
      * Intel® UHD Graphics
      * Intel® Iris® Pro Graphics
      * Intel® Iris® Xe Graphics
      * Intel® Iris® Xe Max Graphics
      * Intel® Arc ™ GPU Series
      * Intel® Data Center GPU Flex Series
      * Intel® Data Center GPU Max Series

   .. tab-item:: Operating Systems for GPU

      * Ubuntu* 22.04 long-term support (LTS), 64-bit
      * Ubuntu* 20.04 long-term support (LTS), 64-bit
      * Windows* 10, 64-bit
      * Windows* 11, 64-bit
      * Centos 7
      * Red Hat Enterprise Linux* 8, 64-bit



   .. tab-item:: 

      * Intel® NPU
      * Intel® Gaussian & Neural Accelerator


NPU and GNA 
#############################

.. tab-set::

   .. tab-item:: Operating Systems for NPU

      * Ubuntu* 22.04 long-term support (LTS), 64-bit
      * Windows* 11, 64-bit  

   .. tab-item:: Operating Systems for GNA

      * Ubuntu* 22.04 long-term support (LTS), 64-bit
      * Ubuntu* 20.04 long-term support (LTS), 64-bit
      * Windows* 10, 64-bit
      * Windows* 11, 64-bit  


Operating systems and developer environment
#######################################################

.. tab-set::

   .. tab-item:: Linux

      * Ubuntu 22.04 with Linux kernel 5.15+  
      * Ubuntu 20.04 with Linux kernel 5.15+  
      * RHEL 8 with Linux kernel 5.4  

      Build environment components:

      * Python* 3.8-3.11
      * Intel® HD Graphics Driver. Required for inference on GPU.
      * GNU Compiler Collection and CMake are needed for building from source:

        * GNU Compiler Collection (GCC)*  7.5 and above
        * CMake* 3.10 or higher  

      Higher versions of kernel might be required for 10th Gen Intel® Core™ Processors,
      11th Gen Intel® Core™ Processors, 11th Gen Intel® Core™ Processors S-Series Processors,
      12th Gen Intel® Core™ Processors, 13th Gen Intel® Core™ Processors,  Intel® Core™ Ultra
      Processors, or 4th Gen Intel® Xeon® Scalable Processors to support CPU, GPU, GNA or
      hybrid-cores CPU capabilities.

   .. tab-item:: Windows

      * Windows 10
      * Windows 11

      Build environment components:

      * Microsoft Visual Studio* 2019
      * CMake 3.10 or higher
      * Python* 3.8-3.11
      * Intel® HD Graphics Driver (Required only for GPU). 

   .. tab-item:: macOS

      * macOS 10.15 and above

      Build environment components:

      * Xcode* 10.3
      * Python 3.8-3.11
      * CMake 3.10 or higher 

   .. tab-item:: DL frameworks versions:

      * TensorFlow* 1.15, 2.12
      * MxNet* 1.9.0 
      * ONNX* 1.14.1 
      * PaddlePaddle* 2.4




.. note::

   OpenVINO Python binaries and binaries on Windows/CentOS7/MACOS(x86) are built
   with oneTBB libraries. Other binaries on Ubuntu and Redhat OSes are built with
   legacy TBB which is released by OS distribution. OpenVINO can be built with 
   either oneTBB or legacy TBB by the user on all OS systems listed. System 
   compatibility and performance are improved on Hybrid CPUs, 
   such as 12th Gen Intel Core and above.



@endsphinxdirective