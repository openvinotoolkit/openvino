# Create a Yocto Image with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_installing_openvino_yocto}

This document provides instructions for creating a Yocto image with Intel® Distribution of OpenVINO™ toolkit.

## System Requirements

Follow the [Yocto Project official documentation](https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html#compatible-linux-distribution) to set up and configure your host machine to be compatible with BitBake.

## Step 1: Set Up Environment

1. Clone the repositories.
   ```sh
   git clone https://git.yoctoproject.org/git/poky --branch langdale
   git clone https://git.yoctoproject.org/meta-intel --branch langdale
   git clone https://git.openembedded.org/meta-openembedded --branch langdale
   git clone https://github.com/kraj/meta-clang.git
   ```

2. Set up the OpenEmbedded build environment.
   ```sh
   source poky/oe-init-build-env
   ```

3. Add BitBake layers.
   ```sh
   bitbake-layers add-layer ../meta-intel
   bitbake-layers add-layer ../meta-openembedded/meta-oe
   bitbake-layers add-layer ../meta-openembedded/meta-python
   bitbake-layers add-layer ../meta-clang
   ```

4. Verify if layers were added (optional step).
   ```sh
   bitbake-layers show-layers
   ```

5. Set up BitBake configurations.
   Include extra configuration in the `conf/local.conf` file in your build directory as required.
   ```sh
   # Build with SSE4.2, AVX2 etc. extensions
   MACHINE = "intel-skylake-64"

   # Enable clDNN GPU plugin when needed.
   # This requires meta-clang and meta-oe layers to be included in bblayers.conf
   # and is not enabled by default.
   # Compute-runtime does not currently support building with LLVM 15 (which is
   # the default in meta-clang master) so enabling GPU plugin may result in
   # build failures.
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

Run BitBake to build your image with OpenVINO packages. For example, to build the minimal image, run the following command:
```sh
bitbake core-image-minimal
```

> **NOTE**: For validation/testing/reviewing purposes, you may consider using the `nohup` command and ensure that your vpn/ssh connection remains uninterrupted.

## Step 3: Verify the Yocto Image

Verify that OpenVINO packages were built successfully. Run the following command:
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

## Additional Resources

- [Troubleshooting Guide](@ref yocto-install-issues)
- [Yocto Project](https://docs.yoctoproject.org/) - official documentation webpage
- [BitBake Tool](https://docs.yoctoproject.org/bitbake/)
- [Poky](https://git.yoctoproject.org/poky)
- [Meta-intel](https://git.yoctoproject.org/meta-intel/tree/README)
- [Meta-openembedded](http://cgit.openembedded.org/meta-openembedded/tree/README)
- [Meta-clang](https://github.com/kraj/meta-clang/tree/master/#readme)
