Quantization-aware Training (QAT)
=================================


Introduction
####################

Quantization-aware Training is a popular method that allows quantizing a model and applying fine-tuning to restore accuracy
degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to
apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are
knowledgeable in Python programming and familiar with the training code for the model in the source DL framework.

Steps required to apply QAT to the model:

.. note::
   Currently, NNCF for TensorFlow supports the optimization of models created using the Keras
   `Sequential API <https://www.tensorflow.org/guide/keras/sequential_model>`__ or
   `Functional API <https://www.tensorflow.org/guide/keras/functional>`__.

1. Apply Post Training Quantization to the Model
#################################################

Quantize the model using the :doc:`Post-Training Quantization <../quantizing-models-post-training/basic-quantization-flow>` method.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [quantize]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [quantize]

2. Fine-tune the Model
#######################

This step assumes applying fine-tuning to the model the same way it is done for the
baseline model. For QAT, it is required to train the model for a few epochs with a small
learning rate, for example, 1e-5. Quantized models perform all computations in the
floating-point precision during fine-tuning by modeling quantization errors in both
forward and backward passes.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [tune_model]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [tune_model]

.. note::
   The precision of weight transitions to INT8 only after converting the model to OpenVINO
   Intermediate Representation. You can expect a reduction in the model footprint only for
   that format.

These steps outline the basics of applying the QAT method from the NNCF. However, in some cases, it is required to save/load model
checkpoints during training. Since NNCF wraps the original model with its own object, it provides an API for these needs.

3. (Optional) Save Checkpoint
##############################

To save a model checkpoint, use the following API:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [save_checkpoint]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [save_checkpoint]

4. (Optional) Restore from Checkpoint
######################################

To restore the model from checkpoint, use the following API:

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
         :language: python
         :fragment: [load_checkpoint]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_tf.py
         :language: python
         :fragment: [load_checkpoint]

Deploying the Quantized Model
##############################

You can convert the model to OpenVINO IR, if needed, compile it and run with OpenVINO without
any additional steps.

.. tab-set::

   .. tab-item:: PyTorch
      :sync: pytorch

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch.py
         :language: python
         :fragment:  [inference]

   .. tab-item:: TensorFlow 2
      :sync: tensorflow-2

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py
         :language: python
         :fragment:  [inference]

For more details, see the corresponding :doc:`documentation <../../running-inference>`.

Examples
#########

* `Quantization-aware Training of Resnet18 PyTorch Model <https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/torch/resnet18>`__
* `Quantization-aware Training of STFPM PyTorch Model <https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/torch/anomalib>`__
* `Quantization-aware Training of MobileNet v2 TensorFlow Model <https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/tensorflow/mobilenet_v2>`__
