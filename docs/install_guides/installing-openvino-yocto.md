# Create a Yocto Image with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_installing_openvino_yocto}
This document provides instructions for creating a Yocto image with Intel® Distribution of OpenVINO™ toolkit.

## System Requirements
Use the [Yocto Project official documentation](https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html#compatible-linux-distribution) to set up and configure your host machine to be compatible with BitBake.

## Step 1: Set Up Environment

### Set Up Git Repositories
The following Git repositories are required to build a Yocto image:

- [Poky](https://git.yoctoproject.org/poky)
- [Meta-intel](https://git.yoctoproject.org/meta-intel/tree/README)
- [Meta-openembedded](http://cgit.openembedded.org/meta-openembedded/tree/README)
- <a href="https://github.com/kraj/meta-clang/blob/master/README.md">Meta-clang</a>

Clone these Git repositories to your host machine: 
```sh
git clone https://git.yoctoproject.org/git/poky --branch kirkstone
git clone https://git.yoctoproject.org/git/meta-intel --branch kirkstone
git clone https://git.openembedded.org/meta-openembedded --branch kirkstone
git clone https://github.com/kraj/meta-clang.git --branch kirkstone-clang12
```

### Set up BitBake Layers

```sh
source poky/oe-init-build-env
bitbake-layers add-layer ../meta-intel
bitbake-layers add-layer ../meta-openembedded/meta-oe
bitbake-layers add-layer ../meta-openembedded/meta-python
bitbake-layers add-layer ../meta-clang
```

### Set up BitBake Configurations

Include extra configuration in `conf/local.conf` in your build directory as required.

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

## Step 2: Build a Yocto Image with OpenVINO Packages

Run BitBake to build your image with OpenVINO packages. To build the minimal image, for example, run:
```sh
bitbake core-image-minimal
```

## Step 3: Verify the Yocto Image with OpenVINO Packages

Verify that OpenVINO packages were built successfully.
Run the following command:
```sh
oe-pkgdata-util list-pkgs | grep openvino
```

If the image was built successfully, it will return the list of packages as below:
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
