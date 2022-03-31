# Using Quantization-aware Training (QAT) {#qat_introduction}

# Introduction
Quantization-aware Training is a popular method that allows quantizing a model and apply fine-tuning to restore accuracy degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are knowledgable in Python* programming and familiar with the training code for the model in the source DL framework.

