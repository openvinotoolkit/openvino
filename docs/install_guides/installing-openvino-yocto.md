# Create a Yocto* Image with OpenVINO™ toolkit {#openvino_docs_install_guides_installing_openvino_yocto}
This document provides instructions for creating a Yocto* image with OpenVINO™ toolkit.

Instructions were validated and tested for [Yocto OpenVINO 2020.3 release](http://git.yoctoproject.org/cgit/cgit.cgi/meta-intel).

## System Requirements
Use the [Yocto Project* official documentation](https://www.yoctoproject.org/docs/latest/mega-manual/mega-manual.html#brief-compatible-distro) to set up and configure your host machine to be compatible with BitBake*.

## Setup 

### Set up Git repositories
The following Git repositories are required to build a Yocto image:

- [Poky](https://www.yoctoproject.org/docs/latest/mega-manual/mega-manual.html#poky)
- [Meta-intel](http://git.yoctoproject.org/cgit/cgit.cgi/meta-intel/tree/README)
- [Meta-openembedded](http://cgit.openembedded.org/meta-openembedded/tree/README)
- <a href="https://github.com/kraj/meta-clang/blob/master/README.md">Meta-clang</a>

Clone these Git repositories to your host machine: 
```sh
git clone https://git.yoctoproject.org/git/poky
git clone https://git.yoctoproject.org/git/meta-intel
git clone https://git.openembedded.org/meta-openembedded
git clone https://github.com/kraj/meta-clang.git
```

### Set up BitBake* Layers

```sh
source poky/oe-init-build-env
bitbake-layers add-layer ../meta-intel
bitbake-layers add-layer ../meta-openembedded/meta-oe
bitbake-layers add-layer ../meta-openembedded/meta-python
bitbake-layers add-layer ../meta-clang
```

### Set up BitBake Configurations

Include extra configuration in conf/local.conf in your build directory as required.

```sh
# Build with SSE4.2, AVX2 etc. extensions
MACHINE = "intel-skylake-64"

# Enable clDNN GPU plugin when needed.
# This requires meta-clang and meta-oe layers to be included in bblayers.conf
# and is not enabled by default.
PACKAGECONFIG_append_pn-openvino-inference-engine = " opencl"

# Enable building inference engine python API.
# This requires meta-python layer to be included in bblayers.conf.
PACKAGECONFIG_append_pn-openvino-inference-engine = " python3"

# This adds inference engine related libraries in the target image.
CORE_IMAGE_EXTRA_INSTALL_append = " openvino-inference-engine"

# This adds inference engine samples in the target image.
CORE_IMAGE_EXTRA_INSTALL_append = " openvino-inference-engine-samples"

# Include inference engine python API package in the target image.
CORE_IMAGE_EXTRA_INSTALL_append = " openvino-inference-engine-python3"

# This adds inference engine unit tests in the target image.
CORE_IMAGE_EXTRA_INSTALL_append = " openvino-inference-engine-ptest"

# Enable MYRIAD plugin
CORE_IMAGE_EXTRA_INSTALL_append = " openvino-inference-engine-vpu-firmware"

# Include model optimizer in the target image.
CORE_IMAGE_EXTRA_INSTALL_append = " openvino-model-optimizer"
```

## Build a Yocto Image with OpenVINO Packages

Run BitBake to build the minimal image with OpenVINO packages: 
```sh
bitbake core-image-minimal
```

## Verify the Created Yocto Image with OpenVINO Packages

Verify that OpenVINO packages were built successfully.
Run 'oe-pkgdata-util list-pkgs | grep openvino' command.
```sh
oe-pkgdata-util list-pkgs | grep openvino
```

Verify that it returns the list of packages below:
```sh
openvino-inference-engine
openvino-inference-engine-dbg
openvino-inference-engine-dev
openvino-inference-engine-ptest
openvino-inference-engine-python3
openvino-inference-engine-samples
openvino-inference-engine-src
openvino-inference-engine-staticdev
openvino-inference-engine-vpu-firmware
openvino-model-optimizer
openvino-model-optimizer-dbg
openvino-model-optimizer-dev
```
