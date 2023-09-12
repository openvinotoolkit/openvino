# Weight Compression {#weight_compression}

@sphinxdirective

Enhancing Model Efficiency with Weight Compression
##################################################################

Weight compression aims to reduce the memory footprint of a model. It can also lead to significant performance improvement for large memory-bound models, such as Large Language Models (LLMs). LLMs and other models, which require extensive memory to store the weights during inference, can benefit from weight compression in the following ways: 

- enabling the inference of exceptionally large models that cannot be accommodated in the memory of the device; 
- improving the inference performance of the models by reducing the latency of the memory access when computing the operations with weights, for example, Linear layers.

Currently, NNCF provides 8-bit weight quantization as a compression method which is basically aimed at optimizing LLM. The main difference between weights compression and full model quantization (post-training quantization) is that activations remain floating-point in the case of weights compression which leads to a better accuracy. On the other hand, weight compression for LLM provides a solid inference performance improvement which is on par with the performance of the full model quantization. In addition, weight compression is data-free and does not require a calibration dataset so it is very easy to use.

Compress Model Weights
######################

The code snippet below shows how to compress the weights of the model represented in OpenVINO IR using NNCF:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino
      
      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_8bit]

Now, the model is ready for compilation and inference. It can be also saved into a compressed format, resulting in a smaller binary file.

Additional Resources
####################

- :doc:`Post-training Quantization <ptq_introduction>`
- :doc:`Training-time Optimization <tmo_introduction>`

@endsphinxdirective
