# Create a Yocto Image with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_installing_openvino_yocto}

This document provides instructions for creating a Yocto image with Intel® Distribution of OpenVINO™ toolkit.

## System Requirements

@sphinxdirective
.. tab:: Supported Linux Distributions

   +--------------------------------+----------------------------------------+
   | Operating System               | Version                                |
   +================================+========================================+
   | Ubuntu                         | 18.04 (LTS), 20.04 (LTS), 22.04 (LTS)  |
   +--------------------------------+----------------------------------------+
   | Fedora                         | 34, 35                                 |
   +--------------------------------+----------------------------------------+
   | AlmaLinux                      | 8.5                                    |
   +--------------------------------+----------------------------------------+
   | Debian GNU/Linux               | 10.x (Buster), 11.x (Bullseye)         |
   +--------------------------------+----------------------------------------+
   | OpenSUSE Leap                  | 15.3                                   |
   +--------------------------------+----------------------------------------+

.. tab::  Required Software

   +---------------------------------------------+---------------------------+
   | Software                                    | Version                   |
   +=============================================+===========================+
   | `GIT <https://git-scm.com/>`__              | 1.8.3.1 or greater        |
   +---------------------------------------------+---------------------------+
   | `tar <https://www.gnu.org/software/tar/>`__ | 1.28 or greater           |
   +---------------------------------------------+---------------------------+
   | `Python <https://www.python.org/>`__        | 3.6 or greater            |
   +---------------------------------------------+---------------------------+
   | `gcc <https://gcc.gnu.org/index.html>`__    | 7.5 or greater            |
   +---------------------------------------------+---------------------------+

@endsphinxdirective


Use the [Yocto Project official documentation](https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html#compatible-linux-distribution) to set up and configure your host machine to be compatible with BitBake.

## Set Up Environment

### 1. Clone the repository.

```sh
git clone https://git.yoctoproject.org/git/poky
```

### 2. Navigate to the "poky" folder and clone the following repositories.

```sh
cd poky
git clone https://git.yoctoproject.org/meta-intel/
git clone https://git.openembedded.org/meta-openembedded
git clone https://github.com/kraj/meta-clang.git
```

### 3. Set up the OpenEmbedded build environment.

```sh
source oe-init-build-env
```

### 4. Add BitBake layers.

```sh
bitbake-layers add-layer ../meta-intel
bitbake-layers add-layer ../meta-openembedded/meta-oe
bitbake-layers add-layer ../meta-openembedded/meta-python
bitbake-layers add-layer ../meta-clang
```

### 5. Verify if layers were added (optional step).

```sh
bitbake-layers show-layers
```

### 6. Set up BitBake configurations.

Include extra configuration in the `conf/local.conf` file in your build directory as required.

```sh
# Build with SSE4.2, AVX2 etc. extensions
MACHINE = "intel-skylake-64"

# Enable clDNN GPU plugin when needed.
# This requires meta-clang and meta-oe layers to be included in bblayers.conf
# and is not enabled by default.
PACKAGECONFIG:append:pn-openvino-inference-engine = " opencl"

# Enable building OpenVINO Python API.
# This requires meta-python layer to be included in bblayers.conf.
PACKAGECONFIG:append:pn-openvino-inference-engine = " python3"

# This adds OpenVINO related libraries in the target image.
CORE_IMAGE_EXTRA_INSTALL:append = " openvino-inference-engine"

# This adds OpenVINO samples in the target image.
CORE_IMAGE_EXTRA_INSTALL:append = " openvino-inference-engine-samples"

# Include OpenVINO Python API package in the target image.
CORE_IMAGE_EXTRA_INSTALL:append = " openvino-inference-engine-python3"

# Enable MYRIAD plugin
CORE_IMAGE_EXTRA_INSTALL:append = " openvino-inference-engine-vpu-firmware"

# Include Model Optimizer in the target image.
CORE_IMAGE_EXTRA_INSTALL:append = " openvino-model-optimizer"
```

### 7. Build a Yocto image with OpenVINO packages.

To build your image with OpenVINO packages, run the following command:

```sh
bitbake core-image-minimal
```

> **NOTE**: For validation/testing/reviewing purposes, you may consider using `nohup` command and ensuring that your vpn/ssh connection remains uninterrupted.

## 8. Verify the Yocto Image with OpenVINO packages.

Verify that OpenVINO packages were built successfully.
Run the following command:
```sh
oe-pkgdata-util list-pkgs | grep openvino
```

If the image build is successful, it will return the list of packages as below:
```sh
openvino-inference-engine
openvino-inference-engine-dbg
openvino-inference-engine-dev
openvino-inference-engine-python3
openvino-inference-engine-samples
openvino-inference-engine-src
openvino-inference-engine-vpu-firmware
openvino-model-optimizer
openvino-model-optimizer-dbg
openvino-model-optimizer-dev
```
## Troubleshooting

When using the `bitbake-layers add-layer meta-intel` command, the following error might occur:
```sh
NOTE: Starting bitbake server...
ERROR: The following required tools (as specified by HOSTTOOLS) appear to be unavailable in PATH, please install them in order to proceed: chrpath diffstat pzstd zstd
```

To resolve the issue, install the `chrpath diffstat zstd` tools:

```sh
sudo apt-get install chrpath diffstat zstd
```

## Additional Resources

- [Yocto Project](https://docs.yoctoproject.org/) - official documentation webpage
- [BitBake Tool](https://docs.yoctoproject.org/bitbake/)
- [Poky](https://git.yoctoproject.org/poky)
- [Meta-intel](https://git.yoctoproject.org/meta-intel/tree/README)
- [Meta-openembedded](http://cgit.openembedded.org/meta-openembedded/tree/README)
- [Meta-clang](https://github.com/kraj/meta-clang/tree/master/#readme)