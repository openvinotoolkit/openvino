# System Requirements {#system_requirements}

@sphinxdirective


Certain hardware requires specific drivers to work properly with OpenVINO. 
These drivers, including Linux* kernels, might require updates to your operating system,
which is not part of OpenVINO installation. Refer to your hardware's documentation 
for updating instructions.


Hardware
##########

.. tab-set::

   .. tab-item:: CPU

      * 6th to 13th generation Intel® Core™ processors
      * 1st to 4th generation Intel® Xeon® Scalable processors
      * ARM and ARM64 CPUs

   .. tab-item:: integrated GPU

      * Intel® HD Graphics 
      * Intel® UHD Graphics 
      * Intel® Iris® Pro Graphics 
      * Intel® Iris® Xe Graphics 
      * Intel® Iris® Xe MAX Graphics

      Not all Intel CPUs include the integrated graphics processor. 
      See `Product Specifications <https://ark.intel.com/>`__
      for information about your processor. 

      Ubuntu 20.04 is not recommended for inference with discrete graphics cards.
      Although this OpenVINO release does support it, the support is limited
      due to driver issues. 

   .. tab-item:: discreet GPU

      * Intel® Data Center GPU Max Series
      * Intel® Data GPU Flex Series Center
      * Intel® Arc ™ GPU

   .. tab-item:: Low-power processing units

      * Intel® NPU
      * Intel® Gaussian & Neural Accelerator


Software 
################

.. tab-set::

   .. tab-item:: Operating System

      * Ubuntu 22.04 long-term support (LTS), 64 bit (Kernel 5.15+)
      * Ubuntu 20.04 LTS, 64 bit (Kernel 5.15+)
      * Ubuntu 18.04 LTS with limitations, 64 bit (Kernel 5.4+)
      * Debian 9 ARM64 and ARM
      * Centos7 64 bit
      * Windows® 10 and 11
      * macOS 10.15+ (x84_64) and 11.0+ (arm64), 64 bit
      * Red Hat Enterprise Linux 8, 64 bit

      Newer versions of the Ubuntu operating system kernel may be required for 10th and 11th 
      generation Intel Core processors, 11th generation Intel Core processors S-Series, 12th
      and 13th generation Intel Core processors, or 4th generation Intel Xeon Scalable 
      processors to support a CPU, GPU, Intel GNA, or hybrid-core with CPU capabilities.

   .. tab-item:: Additional system requirements

      * Python 3.8-3.11
      * GCC 7.5+
      * Cmake 64 bit - 3.10 (Linux),

      Recommended versions for OpenCL™ Drivers:

      * 22.43 for Ubuntu 22.04
      * 22.41 for Ubuntu 20.04
      * 22.28 for Red Hat Enterprise Linux 8 

   .. tab-item:: Supported Framework Versions

      * Apache MxNet 1.9.0
      * ONNX* (Open Neural Network Exchange) 1.14.1 (NEW in 2023.2)
      * PaddlePaddle* 2.4

      remaining :doc:`supported frameworks <openvino_docs_model_processing_introduction>` 
      should work properly regardless of their versions.


.. note::

   OpenVINO Python binaries and binaries on Windows/CentOS7/MACOS(x86) are built
   with oneTBB libraries. Other binaries on Ubuntu and Redhat OSes are built with
   legacy TBB which is released by OS distribution. OpenVINO can be built with 
   either oneTBB or legacy TBB by the user on all OS systems listed. System 
   compatibility and performance are improved on Hybrid CPUs, 
   such as 12th Gen Intel Core and above.



@endsphinxdirective