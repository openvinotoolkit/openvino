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
      * Red Hat Enterprise Linux (RHEL) 8 and 9, 64-bit
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
      * Red Hat Enterprise Linux (RHEL) 8 and 9, 64-bit

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

      * These Accelerators require :doc:`drivers <../../get-started/install-openvino/configurations/configurations-intel-npu>`
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

      * Python 3.9-3.12
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__
        required for inference on GPU
      * GNU Compiler Collection and CMake are needed for building from source:

        * `GNU Compiler Collection (GCC) <https://www.gnu.org/software/gcc/>`__ 7.5 and above
        * `CMake <https://cmake.org/download/>`__ 3.13 or higher

      Higher versions of kernel might be required for 10th Gen Intel® Core™ Processors and above,
      Intel® Core™ Ultra Processors, 4th Gen Intel® Xeon® Scalable Processors and above
      to support CPU, GPU, NPU or hybrid-cores CPU capabilities.

   .. tab-item:: Windows 10 and 11

      OpenVINO Runtime requires certain C++ libraries to operate. To execute ready-made apps,
      the libraries distributed by `Visual Studio redistributable package <https://aka.ms/vs/17/release/vc_redist.x64.exe>`__
      are suggested. For development and compilation of OpenVINO-integrated apps, the build
      environment components are required instead.

      Build environment components:

      * `Microsoft Visual Studio 2019 or later <https://visualstudio.microsoft.com/downloads/>`__
      * `CMake <https://cmake.org/download/>`__ 3.16 or higher
      * `Python <https://www.python.org/downloads/>`__ 3.9-3.12
      * `Intel® HD Graphics Driver <https://downloadcenter.intel.com/product/80939/Graphics-Drivers>`__
        required for inference on GPU

   .. tab-item:: macOS

      * macOS 12.6 and above

      Build environment components:

      * `Xcode <https://developer.apple.com/xcode/>`__ 10.3
      * `CMake <https://cmake.org/download/>`__ 3.13 or higher
      * `Python <https://www.python.org/downloads/>`__ 3.9-3.12

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

OpenVINO Distributions
######################

Different OpenVINO distributions may support slightly different sets of features.
Read installation guides for particular distributions for more details.
Refer to the :doc:`OpenVINO Release Policy <../../../about-openvino/release-notes-openvino/release-policy>`
to learn more about the release types.


.. tab-set::

   .. tab-item:: Archive
      :name: archive-sysreq

      .. tab-set::

         .. tab-item:: Linux
            :name: archive-lnx-sysreq

            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`__
            * GCC:

              .. tab-set::

                 .. tab-item:: Ubuntu
                    :sync: ubuntu

                    * GCC 9.3.0 (for Ubuntu 20.04), GCC 11.3.0 (for Ubuntu 22.04) or GCC 13.2.0 (for Ubuntu 24.04)

                 .. tab-item:: RHEL 8
                    :sync: rhel-8

                    * GCC 8.4.1

                 .. tab-item:: CentOS 7
                    :sync: centos-7

                    * GCC 8.3.1

                      Use the following instructions to install it:

                      Install GCC 8.3.1 via devtoolset-8

                      .. code-block:: sh

                         sudo yum update -y && sudo yum install -y centos-release-scl epel-release
                         sudo yum install -y devtoolset-8

                      Enable devtoolset-8 and check current gcc version

                      .. code-block:: sh

                         source /opt/rh/devtoolset-8/enable
                         gcc -v

         .. tab-item:: macOS
            :name: archive-mac-sysreq

            * `CMake 3.13 or higher <https://cmake.org/download/>`__ (choose "macOS 10.13 or later"). Add ``/Applications/CMake.app/Contents/bin`` to path (for default install).
            * `Python 3.9 - 3.12 <https://www.python.org/downloads/mac-osx/>`__ (choose 3.9 - 3.12). Install and add to path.
            * Apple Xcode Command Line Tools. In the terminal, run ``xcode-select --install`` from any directory
            * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)

         .. tab-item:: Windows
            :name: archive-win-sysreq

            * `C++ libraries (included in Visual Studio redistributable) <https://aka.ms/vs/17/release/vc_redist.x64.exe>`__ (a core dependency for OpenVINO Runtime)
            * `Microsoft Visual Studio 2019 or later <http://visualstudio.microsoft.com/downloads/>`__ (for development and app compilation with OpenVINO)
            * `CMake 3.14 or higher, 64-bit <https://cmake.org/download/>`__ (optional, only required for building sample applications)
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/windows/>`__

            .. note::

               To install Microsoft Visual Studio, follow the `Microsoft Visual Studio installation guide <https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022>`__.
               You can choose to download the Community version. During installation in the **Workloads** tab, choose **Desktop development with C++**.

            .. note::

               You can either use `cmake<version>.msi` which is the installation wizard or `cmake<version>.zip` where you have to go into the `bin` folder and then manually add the path to environmental variables.

            .. important::

               When installing Python, make sure you click the option **Add Python 3.x to PATH** to `add Python <https://docs.python.org/3/using/windows.html#installation-steps>`__ to your `PATH` environment variable.

   .. tab-item:: APT
      :sync: apt-sysreq

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
            * GCC 9.3.0 (for Ubuntu 20.04), GCC 11.3.0 (for Ubuntu 22.04) or GCC 13.2.0 (for Ubuntu 24.04)
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`__

   .. tab-item:: Homebrew
      :name: homebrew-sysreq

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            * `Homebrew <https://brew.sh/>`_
            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
            * GCC 9.3.0 (for Ubuntu 20.04), GCC 11.3.0 (for Ubuntu 22.04) or GCC 13.2.0 (for Ubuntu 24.04)
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`__

         .. tab-item:: macOS
            :sync: macos

            * `Homebrew <https://brew.sh/>`_
            * `CMake 3.13 or higher <https://cmake.org/download/>`__ (choose "macOS 10.13 or later"). Add ``/Applications/CMake.app/Contents/bin`` to path (for default installation).
            * `Python 3.9 - 3.12 <https://www.python.org/downloads/mac-osx/>`__ . Install and add it to path.
            * Apple Xcode Command Line Tools. In the terminal, run ``xcode-select --install`` from any directory to install it.
            * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)

   .. tab-item:: npm
      :name: npm-sysreq

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            All x86_64 / arm64 architectures are supported.

            * `Node.js version 21.0.0 and higher <https://nodejs.org/en/download/package-manager>`__

         .. tab-item:: macOS
            :sync: macos

            All x86_64 / arm64 architectures are supported, however, only for CPU inference.

            * `Node.js version 21.0.0 and higher <https://nodejs.org/en/download/package-manager>`__

         .. tab-item:: Windows
            :sync: Windows

            All x86_64 architectures are supported. Windows ARM is not supported.

            * `Node.js version 21.0.0 and higher <https://nodejs.org/en/download/package-manager/>`__

   .. tab-item:: YUM
      :name: yum-sysreq

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            OpenVINO RPM packages are compatible with and can be run on the following operating systems:

            * RHEL 8.2 and higher
            * Amazon Linux 2022 and 2023
            * Rocky Linux 8.7, 8.8 and 9.2-9.3
            * Alma Linux 8.7, 8.8 and 9.2-9.4
            * Oracle Linux 8.7, 8.8 and 9.2-9.4
            * Fedora 29 and higher up to 41
            * OpenEuler 20.03, 22.03, 23.03 and 24.03
            * Anolis OS 8.6 and 8.8
            * CentOS Stream 8 and 9

            Software:

            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
            * GCC 8.4.1
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`_

   .. tab-item:: ZYPPER
      :name: zypper-sysreq

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux

            OpenVINO RPM packages are compatible with and can be run on openSUSE Tumbleweed only.

            Software:

            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`_
            * GCC 8.2.0
            * `Python 3.9 - 3.12, 64-bit <https://www.python.org/downloads/>`_


The claims stated here may not apply to all use cases and setups. See
:doc:`Legal notices and terms of use <../additional-resources/terms-of-use>` for more information.
