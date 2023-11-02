# Weight Compression {#weight_compression}

@sphinxdirective

Enhancing Model Efficiency with Weight Compression
##################################################################

Weight compression aims to reduce the memory footprint of a model. It can also lead to significant performance improvement for large memory-bound models, such as Large Language Models (LLMs). LLMs and other models, which require extensive memory to store the weights during inference, can benefit from weight compression in the following ways: 

- enabling the inference of exceptionally large models that cannot be accommodated in the memory of the device; 
- improving the inference performance of the models by reducing the latency of the memory access when computing the operations with weights, for example, Linear layers.

Currently, `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides weight quantization to 8 and 4-bit integer data types as a compression method primarily designed to optimize LLMs. The main difference between weights compression and full model quantization (post-training quantization) is that activations remain floating-point in the case of weights compression which leads to a better accuracy. Weight compression for LLMs provides a solid inference performance improvement which is on par with the performance of the full model quantization. In addition, weight compression is data-free and does not require a calibration dataset, making it easy to use.

Compress Model Weights
######################

- **8-bit weight quantization** - this method is aimed at accurate optimization of the model which usually leads to significant performance improvements of the Transformer-based models. Models with 8-bit compressed weights are performant on the vast majority of the supported CPU and GPU platforms.

The code snippet below shows how to do 8-bit quantization of the model weights represented in OpenVINO IR using NNCF:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_8bit]

Now, the model is ready for compilation and inference. It can be also saved into a compressed format, resulting in a smaller binary file.

- **4-bit weight quantization** - this method stands for an INT4-INT8 mixed-precision weight quantization where INT4 is considered as a primary precision and INT8 is a backup one. It usually results in a smaller model size and a lower inference latency but accuracy degradation could be higher, depending on the model. The method has several parameters that can provide different performance-accuracy trade-offs after optimization:

  * ``mode`` - there are two modes to choose from: ``INT4_SYM`` - stands for INT4 symmetric weight quantization and results in faster inference and smaller model size, and ``INT4_ASYM`` - INT4 asymmetric weight quantization with variable zero-point for more accurate results

  * ``group_size`` - controls the size of the group of weights that share the same quantization parameters. The smaller the model size the more accurate the optimized model but the larger its footprint and the slower the inference. We recommend using the following group sizes: ``128``, ``64``, ``32`` (``128`` is default value)

  * ``ratio`` - controls the ratio between INT4 and INT8 compressed layers in the model. For example, 0.8 means that 80% of layers will be compressed to INT4 while the rest to INT8 precision.

The example below shows 4-bit weight quantization applied on top of OpenVINO IR:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_4bit]

.. note::

   OpenVINO also supports 4-bit models from Hugging Face `Transformers <https://github.com/huggingface/transformers>`__ library optimized 
   with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. There is no need to do an extra step of model optimization in this case because 
   model conversion will ensure that int4 optimization results are preserved and model inference will benefit from it.
   

Additional Resources
####################

- :doc:`Post-training Quantization <ptq_introduction>`
- :doc:`Training-time Optimization <tmo_introduction>`
- `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__

@endsphinxdirective
