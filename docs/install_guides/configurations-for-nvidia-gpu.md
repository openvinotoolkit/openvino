# Configurations for NVIDIA GPUs with OpenVINO™ {#openvino_docs_install_guides_configurations_for_nvidia_gpu}

@sphinxdirective

.. meta::
   :description: Learn how to provide additional configuration for NVIDIA 
                 GPUs to work with Intel® Distribution of OpenVINO™ toolkit 
                 on your system.


.. _nvidia gpu guide:

OpenVINO™ NVIDIA GPU plugin enables deep neural networks inference on NVIDIA GPUs. 
The plugin uses custom kernels and cuBLAS, cuDNN, cuTENSOR libraries as a backend 
and requires prior configuration on a supported system.


Linux
####################

OpenVINO™ NVIDIA GPU plugin is supported and validated on the following platform:

+------------------------+------------------------+----------------+
| Operating System       | GPU                    | Driver version |
+========================+========================+================+
| Ubuntu* 20.04 (64-bit) | NVIDIA Quadro RTX 4000 | 520.61.05      |
+------------------------+------------------------+----------------+


OpenVINO™ NVIDIA GPU plugin is not included in OpenVINO™ toolkit. In order to use 
the plugin you need to download and `install dependencies <install-dependencies>`__ and then 
`build it from source code <build-the-plugin-with-cmake>`__. Follow the instructions below.


Install Dependencies
++++++++++++++++++++

1. Install one of the following compilers with support of *C++17*:

   * Install *gcc-7* compiler

   .. code-block:: sh

      sudo apt-get update
      sudo apt-get install gcc-7 g++7


   * Install *clang-8* compiler

   .. code-block:: sh

      sudo apt-get update
      sudo apt-get install clang-8 clang++8


2. Install the compatible `NVIDIA driver <http://www.nvidia.com/Download/index.aspx?lang=en-us>`__
3. Install `CUDA 11.8 <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`__

   Make sure to add ``<your_path_to_cuda>/bin/`` in the *PATH* system variable:

   .. code-block:: sh

      export PATH="<your_path_to_cuda>/bin:$PATH"

4. Install `cuDNN 8.6.0 <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`__
5. Install `cuTENSOR 1.6.1 <https://docs.nvidia.com/cuda/cutensor/getting_started.html#installation-and-compilation>`__

Build the plugin with *cmake*
+++++++++++++++++++++++++++++

First, you need to `build OpenVINO™ package from source <https://github.com/openvinotoolkit/openvino/blob/releases/2023/0/docs/dev/build_linux.md>`__.

Then, you can build the plugin, following the instructions below:

1. Clone the *openvino_contrib* repository:

   .. code-block:: sh

      git clone --recurse-submodules --single-branch --branch=2022.3.0 https://github.com/openvinotoolkit/openvino_contrib.git

2. Go to the plugin directory:

   .. code-block:: sh

      cd openvino_contrib/modules/nvidia_plugin

3. Prepare a build folder:

   .. code-block:: sh

      mkdir build && cd build

4. Build the plugin

   You can build the NVIDIA Plugin with either of the two methods:

   .. tab-set::

      .. tab-item:: Using *build.sh*

         1. Setup the following environment variables:

            .. code-block:: sh

               export OPENVINO_HOME=<OpenVINO source directory>
               export OPENVINO_CONTRIB=<OpenVINOContrib packages source directory>
               export OPENVINO_BUILD_PATH=<OpenVINO build directory>

         2. Then, run one of the following commands:

            .. code-block:: sh

               # Run cmake configuration (if necessary) and then build
               ../build.sh --build

            .. code-block:: sh

               # Run cmake configuration
               ../build.sh --setup

            .. code-block:: sh

               # For old build delete old configuration, generate new one and then build
               ../build.sh --rebuild

      .. tab-item:: Using *OpenVINODeveloperPackage*

         Run the following command:

         .. code-block:: sh

            cmake -DOpenVINODeveloperPackage_DIR=<path to OpenVINO package build folder> -DCMAKE_BUILD_TYPE=Release ..
            cmake --build . --target nvidia_gpu -j `nproc`


Build the plugin with *setup.py*
++++++++++++++++++++++++++++++++

If Python is available on the system, the NVIDIA Plugin can be compiled with the ``setup.py`` script.

1. Clone *openvino_contrib* repository:

   .. code-block:: sh

      git clone --recurse-submodules --single-branch --branch=2022.3.0 https://github.com/openvinotoolkit/openvino_contrib.git


2. Go to the plugin directory:

   .. code-block:: sh

      cd openvino_contrib/modules/nvidia_plugin


3. Add the path to the CUDA ``nvcc`` compiler to the ``CUDACXX`` environment variable:

   .. code-block:: sh

      export CUDACXX=<your_path_to_cuda>/cuda-11.8/bin/nvcc

4. Add the path to the CUDA libraries to the ``LD_LIBRARY_PATH`` environment variable like the next (use yours path)

   .. code-block:: sh

      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/bin/nvcc


5. Run setup.py build command as follows.

   .. code-block:: sh

      export NVIDIA_PLUGIN_SRC_ROOT_DIR=</path/to/openvino_contrib>/modules/nvidia_plugin
      python3 ${NVIDIA_PLUGIN_SRC_ROOT_DIR}/wheel/setup.py build

This will automatically download, build OpenVINO and build CUDA Plugin finally. The location of the resulting library file will be like the next.

.. code-block:: sh

   ${NVIDIA_PLUGIN_SRC_ROOT_DIR}/build/temp.linux-x86_64-3.6/deps/openvino/bin/intel64/Debug/lib/libopenvino_nvidia_gpu_plugin.so


Install as python package with `setup.py`
+++++++++++++++++++++++++++++++++++++++++

To install NVIDIA Plugin as a python package, do all the steps except the last one from the `Build the plugin with setup.py <build-the-plugin-with-setup.py>` section.
After that installation could be done by running the ``setup.py install`` command as follows.

.. code-block:: sh

   export OPENVINO_CONTRIB=</path/to/openvino_contrib>
   python3 ${OPENVINO_CONTRIB}/modules/nvidia_plugin/wheel/setup.py install


This command will install dependent openvino package if needed and update it for using with NVIDIA GPU plugin.


Docker support
++++++++++++++++++++

Build docker container
----------------------

First build docker container:

1. Install Docker.

   .. code-block:: sh

      ./docker.sh install
      su $USER # Relogin for current user

2. Download all ``*.deb`` packages for CUDA and put them in one folder.

3. Build a docker container.

   .. code-block:: sh

      CUDA_PACKAGES_PATH=<path to CUDA packages> ./docker.sh build


Build *openvino_nvidia_gpu_plugin* in a docker container
--------------------------------------------------------

In order to build *openvino_nvidia_gpu_plugin* in docker, follow the steps:

1. Enter the docker container:

   .. code-block:: sh

      docker run --gpus all -it openvino/cudaplugin-2022.3 bin/bash

2. Install `cuDNN 8.6.0 <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`__ 
   and `cuTENSOR 1.6.1 <https://docs.nvidia.com/cuda/cutensor/getting_started.html#installation-and-compilation>`__ 
   packages.

3. Build OpenVINO and *openvino_nvidia_gpu_plugin* according to the instructions described in the `Build the plugin with cmake <build-the-plugin-with-cmake>`__ section.

4. Commit all your changes in the container:

   .. code-block:: sh

      docker commit openvino/cudaplugin-2022.3 <name of new image>

Supported Configuration Parameters
++++++++++++++++++++++++++++++++++

Refer to the following configuration parameters supported by the plugin:

* `ov::hint::performance_mode <enumov_1_1hint_1_1PerformanceMode.html#enum-ov-hint-performancemode>`__
* `ov::hint::execution_mode <enumov_1_1hint_1_1ExecutionMode.html#enum-ov-hint-executionmode>`__
* `ov::hint::inference_precision <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gad605a888f3c9b7598ab55023fbf44240>`__
* `ov::num_streams <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gaeeef815df8212c810bfa11a3f0bd8300>`__
* `ov::enable_profiling <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gafc5bef2fc2b5cfb5a0709cfb04346438>`__


Plugin specific parameters
++++++++++++++++++++++++++

* ``ov::nvidia_gpu::operation_benchmark`` - specifies if operation level benchmark should be run for increasing performance of network (``false`` by default)
* ``ov::nvidia_gpu::use_cuda_graph`` - specifies if NVIDIA plugin attempts to use CUDA Graph feature to speed up sequential network inferences (``true`` by default)

In order to take effect, all the parameters must be set before calling `ov::Core::compile_model() <classov_1_1Core.html#doxid-classov-1-1-core-1a46555f0803e8c29524626be08e7f5c5a>`__.

Compile options
++++++++++++++++++++

During the compilation of *openvino_nvidia_gpu_plugin*, you can specify the following options:

* ``-DCUDA_KERNEL_PRINT_LOG=ON`` enables print logs from kernels.

  .. warning::

     Caution is advised when using these options, as they can print too many logs.

* ``-DENABLE_CUDNN_BACKEND_API`` enables cuDNN backend support that may increase performance of convolutions by 20%.
* ``-DCMAKE_CUDA_ARCHITECTURES=<arch_set>``, for example, 
  `-DCMAKE_CUDA_ARCHITECTURES=75 <https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html>`__ 
  overrides the default *CUDA Compute Capabilities* architectures listed in 
  `openvino_contrib/modules/nvidia_plugin/CMakeLists.txt <https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/nvidia_plugin/CMakeLists.txt>`. 

  This option enables you to build the plugin for a specific architecture or an 
  architecture set. Building for less architectures can significantly decrease 
  the size of ``libopenvino_nvidia_gpu_plugin.so``. 

  To check the compute capability of nVidia devices on your system, you may use the following command:

  .. code-block:: sh

     nvidia-smi --query-gpu=compute_cap --format=csv


Supported Layers and Limitations
################################

The plugin supports IRv10 and higher. Refer to the 
`list of supported layers and its limitations <https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/nvidia_plugin/docs/cuda_opset.md>`__.


What’s Next?
####################

* :doc:`GPU Device <openvino_docs_OV_UG_supported_plugins_GPU>`
* :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image <openvino_docs_install_guides_installing_openvino_docker_linux>`
* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective
