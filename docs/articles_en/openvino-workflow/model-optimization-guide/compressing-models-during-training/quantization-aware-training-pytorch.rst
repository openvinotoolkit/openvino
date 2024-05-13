Quantization-aware Training (QAT) with PyTorch
===============================================

Here, we provide the steps that are required to integrate QAT from NNCF into the training script written with
PyTorch:


1. Apply Post Training Quantization to the model
####################################

Quantize the model using the :doc:`Post-Training Quantization <../quantizing-models-post-training/basic-quantization-flow>` method.

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [quantize]


2. Fine-tune the model
########################

This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the
case of QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5.

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [tune_model]



3. Export quantized model
####################################

When fine-tuning finishes, the quantized model can be exported to the ONNX format for further inference.

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [export]


.. note::
   The precision of weights gets INT8 only after the step of model conversion to OpenVINO Intermediate Representation.
   You can expect the model footprint reduction only for that format.


These were the basic steps to applying the QAT method from the NNCF. However, it is required in some cases to save/load model
checkpoints during the training. Since NNCF wraps the original model with its own object it provides an API for these needs.

4. (Optional) Save checkpoint
####################################

To save model checkpoint use the following API:

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [save_checkpoint]


5. (Optional) Restore from checkpoint
################################################

To restore the model from checkpoint you should use the following API:

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [load_checkpoint]


Deploying quantized model
#########################

The quantized model can be deployed with OpenVINO in the same way as the baseline model. No extra steps or options are
required in this case. For more details, see the corresponding :doc:`documentation <../../running-inference>`.

Example
####################

* `Quantizing PyTorch model with NNCF <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training>`__
