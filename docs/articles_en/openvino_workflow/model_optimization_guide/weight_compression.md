# Weight Compression {#weight_compression}

@sphinxdirective

Enhancing Model Efficiency with Weight Compression
##################################################################

Weight compression aims to reduce the memory footprint of a model. It can also lead to significant performance improvement for large memory-bound models, such as Large Language Models (LLMs). LLMs and other models, which require extensive memory to store the weights during inference, can benefit from weight compression in the following ways: 

- enabling the inference of exceptionally large models that cannot be accommodated in the memory of the device; 
- improving the inference performance of the models by reducing the latency of the memory access when computing the operations with weights, for example, Linear layers.

Currently, `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides weight quantization to 8 and 4-bit integer data types as a compression method primarily designed to optimize LLMs. The main difference between weights compression and full model quantization (post-training quantization) is that activations remain floating-point in the case of weight compression, resulting in better accuracy. Weight compression for LLMs provides a solid inference performance improvement which is on par with the performance of the full model quantization. In addition, weight compression is data-free and does not require a calibration dataset, making it easy to use.

Compress Model Weights
######################

- **8-bit weight quantization** - this method is aimed at accurate optimization of the model, which usually leads to significant performance improvements for Transformer-based models. Models with 8-bit compressed weights are performant on the vast majority of supported CPU and GPU platforms.

The code snippet below shows how to do 8-bit quantization of the model weights represented in OpenVINO IR using NNCF:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_8bit]

Now, the model is ready for compilation and inference. It can be also saved into a compressed format, resulting in a smaller binary file.

- **4-bit weight quantization** - this method stands for an INT4-INT8 mixed-precision weight quantization, where INT4 is considered as the primary precision and INT8 is the backup one. It usually results in a smaller model size and lower inference latency, although the accuracy degradation could be higher, depending on the model. The method has several parameters that can provide different performance-accuracy trade-offs after optimization:

  * ``mode`` - there are two modes to choose from: ``INT4_SYM`` - stands for INT4 symmetric weight quantization and results in faster inference and smaller model size, and ``INT4_ASYM`` - INT4 asymmetric weight quantization with variable zero-point for more accurate results.

  * ``group_size`` - controls the size of the group of weights that share the same quantization parameters. Smaller model size results in a more accurate optimized model but with a larger footprint and slower inference. The following group sizes are recommended: ``128``, ``64``, ``32`` (``128`` is default value)

  * ``ratio`` - controls the ratio between INT4 and INT8 compressed layers in the model. For example, 0.8 means that 80% of layers will be compressed to INT4, while the rest will be compressed to INT8 precision.

The example below shows 4-bit weight quantization applied on top of OpenVINO IR:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_4bit]

.. note::

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__ library optimized 
   with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there is no need for an additional model optimization step because model conversion will automatically preserve the INT4 optimization results, allowing model inference to benefit from it.


The table below shows examples of Text Generation models with different optimization settings:

.. list-table::
   :widths: 40 55 25 25
   :header-rows: 1

   * - Model
     - Optimization
     - Perplexity
     - Model Size (Gb)
   * - databricks/dolly-v2-3b
     - FP32
     - 5.01
     - 10.3
   * - databricks/dolly-v2-3b
     - INT8
     - 5.07
     - 2.6
   * - databricks/dolly-v2-3b
     - INT4_ASYM,group_size=32,ratio=0.5
     - 5.28
     - 2.2
   * - facebook/opt-6.7b
     - FP32
     - 4.25
     - 24.8
   * - facebook/opt-6.7b
     - INT8
     - 4.27
     - 6.2
   * - facebook/opt-6.7b
     - INT4_ASYM,group_size=64,ratio=0.8
     - 4.32
     - 4.1
   * - meta-llama/Llama-2-7b-chat-hf
     - FP32
     - 3.28
     - 25.1
   * - meta-llama/Llama-2-7b-chat-hf
     - INT8
     - 3.29
     - 6.3
   * - meta-llama/Llama-2-7b-chat-hf
     - INT4_ASYM,group_size=128,ratio=0.8
     - 3.41
     - 4.0
   * - togethercomputer/RedPajama-INCITE-7B-Instruct
     - FP32
     - 4.15
     - 25.6
   * - togethercomputer/RedPajama-INCITE-7B-Instruct
     - INT8
     - 4.17
     - 6.4
   * - togethercomputer/RedPajama-INCITE-7B-Instruct
     - INT4_ASYM,group_size=128,ratio=1.0
     - 4.17
     - 3.6
   * - meta-llama/Llama-2-13b-chat-hf
     - FP32
     - 2.92
     - 48.5
   * - meta-llama/Llama-2-13b-chat-hf
     - INT8
     - 2.91
     - 12.1
   * - meta-llama/Llama-2-13b-chat-hf
     - INT4_SYM,group_size=64,ratio=0.8
     - 2.98
     - 8.0
   

Additional Resources
####################

- :doc:`Post-training Quantization <ptq_introduction>`
- :doc:`Training-time Optimization <tmo_introduction>`
- `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__

@endsphinxdirective
