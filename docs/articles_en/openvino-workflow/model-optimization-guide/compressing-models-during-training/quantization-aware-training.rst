Quantization-aware Training (QAT)
=================================

.. toctree::
   :maxdepth: 1
   :hidden:

   Quantization-aware Training with PyTorch <quantization-aware-training-pytorch>
   Quantization-aware Training with TensorFlow <quantization-aware-training-tensorflow>

Introduction
####################

Quantization-aware Training is a popular method that allows quantizing a model and applying fine-tuning to restore accuracy
degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to
apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are
knowledgeable in Python programming and familiar with the training code for the model in the source DL framework.

:doc:`Quantization-aware Training with PyTorch <quantization-aware-training-pytorch>`

:doc:`Quantization-aware Training with TensorFlow <quantization-aware-training-tensorflow>`