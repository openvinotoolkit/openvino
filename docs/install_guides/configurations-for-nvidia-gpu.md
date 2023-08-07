# Configurations for NVIDIA GPUs with OpenVINO™ {#openvino_docs_install_guides_configurations_for_nvidia_gpu}

@sphinxdirective

.. meta::
   :description: Learn how to provide additional configuration for NVIDIA 
                 GPUs to work with Intel® Distribution of OpenVINO™ toolkit 
                 on your system.


.. _nvidia gpu guide:

OpenVINO™ NVIDIA GPU plugin enables deep neural networks inference on NVIDIA GPUs. 
The plugin uses custom kernels and cuBLAS, cuDNN, cuTENSOR libraries as a backend 
and requires prior configuration on the following supported operating systems:

* Ubuntu 20.04 long-term support (LTS), 64-bit
* Windows 10, 64-bit
* Windows 11, 64-bit

OpenVINO™ NVIDIA GPU plugin is not included in OpenVINO™ toolkit. If you want to use 
the plugin, first you need to `install dependencies <install-dependencies>`__, and 
then `build it from source code <#whats-next>`__.

Install Dependencies
####################

1. Install the compatible `NVIDIA driver <http://www.nvidia.com/Download/index.aspx?lang=en-us>`__.
2. Install `CUDA 11.8 <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`__.

   Make sure to add ``<your_path_to_cuda>/bin/`` in the *PATH* system variable:

   .. code-block:: sh

      export PATH="<your_path_to_cuda>/bin:$PATH"

3. Install `cuDNN 8.6.0 <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`__.
4. Install `cuTENSOR 1.6.1 <https://docs.nvidia.com/cuda/cutensor/getting_started.html#installation-and-compilation>`__.


Supported Configuration Parameters
##################################

Refer to the following configuration parameters supported by the plugin:

* `ov::hint::performance_mode <enumov_1_1hint_1_1PerformanceMode.html#enum-ov-hint-performancemode>`__
* `ov::hint::execution_mode <enumov_1_1hint_1_1ExecutionMode.html#enum-ov-hint-executionmode>`__
* `ov::hint::inference_precision <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gad605a888f3c9b7598ab55023fbf44240>`__
* `ov::num_streams <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gaeeef815df8212c810bfa11a3f0bd8300>`__
* `ov::enable_profiling <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gafc5bef2fc2b5cfb5a0709cfb04346438>`__

Plugin specific parameters
##########################

* ``ov::nvidia_gpu::operation_benchmark`` - specifies if operation level benchmark should be run for increasing performance of network (``false`` by default)
* ``ov::nvidia_gpu::use_cuda_graph`` - specifies if NVIDIA plugin attempts to use CUDA Graph feature to speed up sequential network inferences (``true`` by default)

In order to take effect, all the parameters must be set before calling `ov::Core::compile_model() <classov_1_1Core.html#doxid-classov-1-1-core-1a46555f0803e8c29524626be08e7f5c5a>`__.

Supported Layers and Limitations
################################

.. hint::

   The plugin supports IRv10 and higher. Refer to the 
   `list of supported layers and its limitations <https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/nvidia_plugin/docs/cuda_opset.md>`__.


What’s Next?
####################

* `Build OpenVINO™ NVIDIA GPU plugin from source code <https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/nvidia_plugin/README.md#build-with-cmake>`__.
* :doc:`Install Intel® Distribution of OpenVINO™ toolkit for Linux from a Docker Image <openvino_docs_install_guides_installing_openvino_docker_linux>`
* `Docker CI framework for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/README.md>`__
* `Get Started with DockerHub CI for Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/get-started.md>`__
* `Dockerfiles with Intel® Distribution of OpenVINO™ toolkit <https://github.com/openvinotoolkit/docker_ci/blob/master/dockerfiles/README.md>`__

@endsphinxdirective
