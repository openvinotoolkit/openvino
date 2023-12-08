.. {#openvino_docs_install_guides_installing_openvino_yocto}

Create a Yocto Image with Intel® Distribution of OpenVINO™ toolkit
=====================================================================


.. meta::
   :description: Learn how to create a Yocto image with Intel® Distribution of 
                 OpenVINO™ toolkit on your host machine.

This document provides instructions for creating a Yocto image with Intel® Distribution of OpenVINO™ toolkit.

Follow the `Yocto Project official documentation <https://docs.yoctoproject.org/brief-yoctoprojectqs/index.html#compatible-linux-distribution>`__ to set up and configure your host machine to be compatible with BitBake.

Step 1: Set Up Environment
##########################

1. Clone the repositories.
   
   .. code-block:: sh
      
      git clone https://git.yoctoproject.org/git/poky
      git clone https://git.yoctoproject.org/meta-intel
      git clone https://git.openembedded.org/meta-openembedded
      git clone https://github.com/kraj/meta-clang.git
   

2. Set up the OpenEmbedded build environment.

   .. code-block:: sh
      
      source poky/oe-init-build-env
   

3. Add BitBake layers.

   .. code-block:: sh
      
      bitbake-layers add-layer ../meta-intel
      bitbake-layers add-layer ../meta-openembedded/meta-oe
      bitbake-layers add-layer ../meta-openembedded/meta-python
      bitbake-layers add-layer ../meta-clang
   

4. Verify if layers were added (optional step).

   .. code-block:: sh
      
      bitbake-layers show-layers
   

5. Set up BitBake configurations.
   Include extra configuration in the `conf/local.conf` file in your build directory as required.

   .. code-block:: sh
      
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
   
      # Include model conversion API in the target image.
      CORE_IMAGE_EXTRA_INSTALL:append = " openvino-model-optimizer"
   

Step 2: Build a Yocto Image with OpenVINO Packages
##################################################

Run BitBake to build your image with OpenVINO packages. For example, to build the minimal image, run the following command:

.. code-block:: sh
   
   bitbake core-image-minimal


.. note:: 
   For validation/testing/reviewing purposes, you may consider using the ``nohup`` command and ensure that your vpn/ssh connection remains uninterrupted.

Step 3: Verify the Yocto Image
##############################

Verify that OpenVINO packages were built successfully. Run the following command:

.. code-block:: sh
   
   oe-pkgdata-util list-pkgs | grep openvino


If the image build is successful, it will return the list of packages as below:

.. code-block:: sh
   
   openvino-inference-engine
   openvino-inference-engine-dbg
   openvino-inference-engine-dev
   openvino-inference-engine-python3
   openvino-inference-engine-samples
   openvino-inference-engine-src
   openvino-model-optimizer
   openvino-model-optimizer-dbg
   openvino-model-optimizer-dev

Additional Resources
####################

- :ref:`Troubleshooting Guide <yocto_install_issues>`
- `Yocto Project <https://docs.yoctoproject.org/>`__ - official documentation webpage
- `BitBake Tool <https://docs.yoctoproject.org/bitbake/>`__
- `Poky <https://git.yoctoproject.org/poky>`__
- `Meta-intel <https://git.yoctoproject.org/meta-intel/tree/README>`__
- `Meta-openembedded <http://cgit.openembedded.org/meta-openembedded/tree/README>`__
- `Meta-clang <https://github.com/kraj/meta-clang/tree/master/#readme>`__



