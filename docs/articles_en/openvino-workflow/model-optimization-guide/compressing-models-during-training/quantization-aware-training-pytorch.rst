Quantization-aware Training (QAT) with PyTorch
===============================================

Below are the steps required to integrate QAT from NNCF into a training script written with
PyTorch:


1. Apply Post Training Quantization to the Model
##################################################

Quantize the model using the :doc:`Post-Training Quantization <../quantizing-models-post-training/basic-quantization-flow>` method.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
   :language: python
   :fragment: [quantize]


2. Fine-tune the Model
########################

This step assumes applying fine-tuning to the model the same way it is done for the baseline model. For QAT, it is required to train the model for a few epochs with a small learning rate, for example, 1e-5.
Quantized models perform all computations in floating-point precision during fine-tuning by modeling quantization errors in both forward and backward passes.

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
   :language: python
   :fragment: [tune_model]


.. note::
   The precision of weights transitions to INT8 only after converting the model to OpenVINO Intermediate Representation.
   You can expect a reduction in model footprint only for that format.


These steps outline the basics of applying the QAT method from the NNCF. However, in some cases, it is required to save/load model
checkpoints during training. Since NNCF wraps the original model with its own object, it provides an API for these needs.

3. (Optional) Save Checkpoint
####################################

To save a model checkpoint, use the following API:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
   :language: python
   :fragment: [save_checkpoint]


4. (Optional) Restore from Checkpoint
################################################

To restore the model from checkpoint, use the following API:

.. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
   :language: python
   :fragment: [load_checkpoint]


Deploying the Quantized Model
###############################

The model can be converted into the OpenVINO Intermediate Representation (IR) if needed, compiled, and run with OpenVINO without any additional steps.

.. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_torch.py
   :language: python
   :fragment:  [inference]

For more details, see the corresponding :doc:`documentation <../../running-inference>`.

Examples
####################

* `Quantization-aware Training of Resnet18 PyTorch Model <https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/torch/resnet18>`__
* `Quantization-aware Training of STFPM PyTorch Model <https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/torch/anomalib>`__
