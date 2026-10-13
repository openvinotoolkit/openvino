---
sidebar_position: 3
sidebar_label: "System Requirements"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# System Requirements

:::note
Certain hardware, including but not limited to GPU and NPU, requires manual installation of
specific drivers and/or other software components to work correctly and/or to utilize
hardware capabilities at their best. This might require updates to the operating
system, including but not limited to Linux kernel, please refer to their documentation
for details. These modifications should be handled by user and are not part of OpenVINO
installation.
:::

## CPU

<Tabs>
  <TabItem value="supported-hardware" label="Supported Hardware">
  - Intel® Core™ Ultra Series 1, Series 2 and Series 3
  - Intel® Xeon® 6 processor
  - Intel Atom® Processor X Series
  - Intel Atom® processor with Intel® SSE4.2 support
  - Intel® Pentium® processor N4200/5, N3350/5, N3450/5 with Intel® HD Graphics
  - 6th - 14th generation Intel® Core™ processors
  - 1st - 5th generation Intel® Xeon® Scalable Processors
  - ARM CPUs with armv7a and higher, ARM64 CPUs with arm64-v8a and higher, Apple® Mac with Apple silicon
  </TabItem>
  <TabItem value="supported-operating-systems" label="Supported Operating Systems">
  - Windows 11, 64-bit
  - Windows 10, 64-bit
  - Ubuntu 26.04 long-term support (LTS), 64-bit (Kernel 7.0+)
  - Ubuntu 24.04 long-term support (LTS), 64-bit (Kernel 6.8+)
  - Ubuntu 22.04 long-term support (LTS), 64-bit (Kernel 5.15+)
  - macOS 12.6 and above, ARM64
  - Red Hat Enterprise Linux (RHEL) 8 and 9, 64-bit
  - openSUSE Tumbleweed, 64-bit and ARM64
  - Ubuntu 22.04 ARM64
  </TabItem>
</Tabs>

## GPU

<Tabs>
  <TabItem value="supported-hardware" label="Supported Hardware">
  - Intel® Arc™ GPU Series
  - Intel® HD Graphics
  - Intel® UHD Graphics
  - Intel® Iris® Pro Graphics
  - Intel® Iris® Xe Graphics
  - Intel® Iris® Xe Max Graphics
  - Intel® Data Center GPU Flex Series
  - Intel® Data Center GPU Max Series
  </TabItem>
  <TabItem value="supported-operating-systems" label="Supported Operating Systems">
  - Windows 11, 64-bit
  - Windows 10, 64-bit
  - Ubuntu 26.04 long-term support (LTS), 64-bit (Kernel 7.0+)
  - Ubuntu 24.04 long-term support (LTS), 64-bit
  - Ubuntu 22.04 long-term support (LTS), 64-bit
  - Red Hat Enterprise Linux (RHEL) 8 and 9, 64-bit
  </TabItem>
  <TabItem value="additional-considerations" label="Additional considerations">
  - The use of GPU requires drivers that are not included in the Intel®
    Distribution of OpenVINO™ toolkit package.
  - Processor graphics are not included in all processors. See
    [Product Specifications](https://ark.intel.com/)
    for information about your processor.
  - OpenCL™ driver versions required may vary, depending on hardware and operating Systems
    used. Consult driver documentation to select the best version for your setup.
  </TabItem>
</Tabs>

## Intel® Neural Processing Unit

<Tabs>
  <TabItem value="operating-systems-for-npu" label="Operating Systems for NPU">
  - Ubuntu 24.04 long-term support (LTS), 64-bit (preview support)
  - Ubuntu 22.04 long-term support (LTS), 64-bit
  - Windows 11, 64-bit (22H2 and later)
  </TabItem>
  <TabItem value="additional-considerations" label="Additional considerations">
  - These Accelerators require [drivers](../../get-started/install-openvino/configurations/configurations-intel-npu.md)
    that are not included in the Intel® Distribution of OpenVINO™ toolkit package.
  - Users can access the NPU plugin through the OpenVINO archives on
    the [download page](../../get-started/install-openvino/index.md).
  </TabItem>
</Tabs>

## Operating systems and developer environment

<Tabs>
  <TabItem value="linux-os" label="Linux OS">
  - Ubuntu 24.04 with Linux kernel 6.8+
  - Ubuntu 22.04 with Linux kernel 5.15+
  - Red Hat Enterprise Linux 9.3-9.4 with Linux kernel 5.4

  Build environment components:

  - Python 3.10-3.14
  - [Intel® HD Graphics Driver](https://downloadcenter.intel.com/product/80939/Graphics-Drivers)
    required for inference on GPU
  - GNU Compiler Collection and CMake are needed for building from source:
    
    - [GNU Compiler Collection (GCC)](https://www.gnu.org/software/gcc/) 7.5 and above
    - [CMake](https://cmake.org/download/) 3.26 or higher

  Higher versions of kernel might be required for 10th Gen Intel® Core™ Processors and above,
  Intel® Core™ Ultra Processors, 4th Gen Intel® Xeon® Scalable Processors and above
  to support CPU, GPU, NPU or hybrid-cores CPU capabilities.
  </TabItem>
  <TabItem value="windows-10-and-11" label="Windows 10 and 11">
  OpenVINO Runtime requires certain C++ libraries to operate. To execute ready-made apps,
  the libraries distributed by [Visual Studio redistributable package](https://aka.ms/vs/17/release/vc_redist.x64.exe)
  are suggested. For development and compilation of OpenVINO-integrated apps, the build
  environment components are required instead.

  Build environment components:

  - [Microsoft Visual Studio 2019 or later](https://visualstudio.microsoft.com/downloads/)
  - [CMake](https://cmake.org/download/) 3.26 or higher
  - [Python](https://www.python.org/downloads/) 3.10-3.14
  - [Intel® HD Graphics Driver](https://downloadcenter.intel.com/product/80939/Graphics-Drivers)
    required for inference on GPU
  </TabItem>
  <TabItem value="macos" label="macOS">
  - macOS 12.6 and above

  Build environment components:

  - [Xcode](https://developer.apple.com/xcode/) 10.3
  - [CMake](https://cmake.org/download/) 3.26 or higher
  - [Python](https://www.python.org/downloads/) 3.10-3.14
  </TabItem>
  <TabItem value="dl-framework-versions" label="DL framework versions:">
  - TensorFlow 1.15.5 - 2.17
  - PyTorch 2.4
  - ONNX 1.16
  - PaddlePaddle 2.6
  - JAX 0.4.31 (via a path of jax2tf with native\_serialization=False)

  This package can be installed on other versions of DL Frameworks
  but only the versions specified here are fully validated.
  </TabItem>
</Tabs>

:::note
OpenVINO Python binaries are built with and redistribute oneTBB libraries.
:::

## OpenVINO Distributions

Different OpenVINO distributions may support slightly different sets of features.
Read installation guides for particular distributions for more details.
Refer to the [OpenVINO Release Policy](../../../about-openvino/release-notes-openvino/release-policy)
to learn more about the release types.

<Tabs>
  <TabItem value="archive" label="Archive">
  <Tabs>
    <TabItem value="linux" label="Linux">
    - [CMake 3.26 or higher, 64-bit](https://cmake.org/download/)
    - [Python 3.10 - 3.14, 64-bit](https://www.python.org/downloads/)
    - GCC:
      
      <Tabs>
        <TabItem value="ubuntu" label="Ubuntu">
        - GCC 11.3.0 (for Ubuntu 22.04), GCC 13.2.0 (for Ubuntu 24.04) or GCC 15.2.0 (for Ubuntu 26.04)
        </TabItem>
        <TabItem value="rhel-8" label="RHEL 8">
        - GCC 8.4.1
        </TabItem>
      </Tabs>
    </TabItem>
    <TabItem value="macos" label="macOS">
    - [CMake 3.26 or higher](https://cmake.org/download/) (choose "macOS 10.13 or later"). Add `/Applications/CMake.app/Contents/bin` to path (for default install).
    - [Python 3.10 - 3.14](https://www.python.org/downloads/mac-osx/) (choose 3.10 - 3.14). Install and add to path.
    - Apple Xcode Command Line Tools. In the terminal, run `xcode-select --install` from any directory
    - (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)
    </TabItem>
    <TabItem value="windows" label="Windows">
    - [C++ libraries (included in Visual Studio redistributable)](https://aka.ms/vs/17/release/vc_redist.x64.exe) (a core dependency for OpenVINO Runtime)
    - [Microsoft Visual Studio 2019 or later](http://visualstudio.microsoft.com/downloads/) (for development and app compilation with OpenVINO)
    - [CMake 3.26 or higher, 64-bit](https://cmake.org/download/) (optional, only required for building sample applications)
    - [Python 3.10 - 3.14, 64-bit](https://www.python.org/downloads/windows/)

    > **Note:**
    >
    > To install Microsoft Visual Studio, follow the [Microsoft Visual Studio installation guide](https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022).
    > You can choose to download the Community version. During installation in the **Workloads** tab, choose **Desktop development with C++**.

    > **Note:**
    >
    > You can either use cmake`<version>`.msi which is the installation wizard or cmake`<version>`.zip where you have to go into the bin folder and then manually add the path to environmental variables.

    > **Important:**
    >
    > When installing Python, make sure you click the option **Add Python 3.x to PATH** to [add Python](https://docs.python.org/3/using/windows.html#installation-steps) to your PATH environment variable.
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="apt-sysreq" label="APT">
  <Tabs>
    <TabItem value="linux" label="Linux">
    - [CMake 3.26 or higher, 64-bit](https://cmake.org/download/)
    - GCC 11.3.0 (for Ubuntu 22.04), GCC 13.2.0 (for Ubuntu 24.04) or GCC 15.2.0 (for Ubuntu 26.04)
    - [Python 3.10 - 3.14, 64-bit](https://www.python.org/downloads/)
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="homebrew" label="Homebrew">
  <Tabs>
    <TabItem value="linux" label="Linux">
    - [Homebrew](https://brew.sh/)
    - [CMake 3.26 or higher, 64-bit](https://cmake.org/download/)
    - GCC 11.3.0 (for Ubuntu 22.04), GCC 13.2.0 (for Ubuntu 24.04) or GCC 15.2.0 (for Ubuntu 26.04)
    - [Python 3.10 - 3.14, 64-bit](https://www.python.org/downloads/)
    </TabItem>
    <TabItem value="macos" label="macOS">
    - [Homebrew](https://brew.sh/)
    - [CMake 3.26 or higher](https://cmake.org/download/) (choose "macOS 10.13 or later"). Add `/Applications/CMake.app/Contents/bin` to path (for default installation).
    - [Python 3.10 - 3.14](https://www.python.org/downloads/mac-osx/) . Install and add it to path.
    - Apple Xcode Command Line Tools. In the terminal, run `xcode-select --install` from any directory to install it.
    - (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="npm" label="npm">
  <Tabs>
    <TabItem value="linux" label="Linux">
    All x86\_64 / arm64 architectures are supported.

    - [Node.js version 21.0.0 and higher](https://nodejs.org/en/download/package-manager)
    </TabItem>
    <TabItem value="macos" label="macOS">
    All x86\_64 / arm64 architectures are supported, however, only for CPU inference.

    - [Node.js version 21.0.0 and higher](https://nodejs.org/en/download/package-manager)
    </TabItem>
    <TabItem value="windows" label="Windows">
    All x86\_64 architectures are supported. Windows ARM is not supported.

    - [Node.js version 21.0.0 and higher](https://nodejs.org/en/download/package-manager/)
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="yum" label="YUM">
  <Tabs>
    <TabItem value="linux" label="Linux">
    OpenVINO RPM packages are compatible with and can be run on the following operating systems:

    - RHEL 8.2 and higher
    - Amazon Linux 2022 and 2023
    - Rocky Linux 8.7, 8.8 and 9.2-9.3
    - Alma Linux 8.7, 8.8 and 9.2-9.4
    - Oracle Linux 8.7, 8.8 and 9.2-9.4
    - Fedora 29 and higher up to 41
    - OpenEuler 20.03, 22.03, 23.03 and 24.03
    - Anolis OS 8.6 and 8.8
    - CentOS Stream 8 and 9

    Software:

    - [CMake 3.26 or higher, 64-bit](https://cmake.org/download/)
    - GCC 8.4.1
    - [Python 3.10 - 3.14, 64-bit](https://www.python.org/downloads/)
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="zypper" label="ZYPPER">
  <Tabs>
    <TabItem value="linux" label="Linux">
    OpenVINO RPM packages are compatible with and can be run on openSUSE Tumbleweed only.

    Software:

    - [CMake 3.26 or higher, 64-bit](https://cmake.org/download/)
    - GCC 8.2.0
    - [Python 3.10 - 3.14, 64-bit](https://www.python.org/downloads/)
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

The claims stated here may not apply to all use cases and setups. See
[Legal notices and terms of use](../additional-resources/terms-of-use.md) for more information.
