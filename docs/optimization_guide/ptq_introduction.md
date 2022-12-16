# Quantizing Models Post-training {#ptq_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   pot_introduction
   nncf_ptq_introduction

@endsphinxdirective

Post-training model optimization is the process of applying special methods that transform the model into a more hardware-friendly representation without retraining or fine-tuning. The most popular and widely-spread method here is 8-bit post-training quantization because it is:
* It is easy-to-use.
* It does not hurt accuracy a lot.
* It provides significant performance improvement.
* It suites many hardware available in stock since most of them support 8-bit computation natively.

8-bit integer quantization lowers the precision of weights and activations to 8 bits, which leads to almost 4x reduction in the model footprint and significant improvements in inference speed, mostly due to lower throughput required for the inference. This lowering step is done offline, before the actual inference, so that the model gets transformed into the quantized representation. The process does not require a training dataset or a training pipeline in the source DL framework. 

![](../img/quantization_picture.png)

To apply post-training methods in OpenVINO&trade;, you need:
* A floating-point precision model, FP32 or FP16, converted into the OpenVINO&trade; Intermediate Representation (IR) format that can be run on CPU.
* A representative calibration dataset, representing a use case scenario, for example, of 300 samples.
* In case of accuracy constraints, a validation dataset and accuracy metrics should be available.

Currently, OpenVINO provides two workflows with post-training quantization capabilities:
* [Post-training Quantization with POT](@ref pot_introduction) - works with models in OpenVINO Intermediate Representation (IR) only.
* [Post-training Quantization with NNCF](@ref nncf_ptq_introduction) - cross-framework solution for model optimization that provides a new simple API for post-training quantization.