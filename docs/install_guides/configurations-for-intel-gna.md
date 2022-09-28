# Configurations for Intel® Gaussian & Neural Accelerator (GNA) with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_configurations_for_intel_gna}

This page introduces additional configurations for Intel® Gaussian & Neural Accelerator (GNA) with Intel® Distribution of OpenVINO™ toolkit on Linux and Windows.

> **NOTE**: On platforms where Intel® GNA is not enabled in the BIOS, the driver cannot be installed, so the GNA plugin uses the software emulation mode only.

### Drivers and Dependencies

Intel® GNA hardware requires a driver to be installed on the system.

@sphinxdirective

.. _gna guide:

@endsphinxdirective

## Linux

### Prerequisites

Ensure that make, gcc, and Linux kernel headers are installed.

### Configuration steps

1. Download [Intel® GNA driver for Ubuntu Linux 18.04.3 LTS (with HWE Kernel version 5.4+)](https://storage.openvinotoolkit.org/drivers/gna/)
2. Run the sample_install.sh script provided in the installation package:
   ```sh
   prompt$ ./scripts/sample_install.sh
   ```

You can also build and install the driver manually by using the following commands:
```sh
prompt$ cd src/
prompt$ make
prompt$ sudo insmod intel_gna.ko
```

To unload the driver:
```sh
prompt$ sudo rmmod intel_gna
```

@sphinxdirective

.. _gna guide windows:

@endsphinxdirective

## Windows

Intel® GNA driver for Windows is available through Windows Update.

