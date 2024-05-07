.. {#qat_introduction}

Quantization-aware Training (QAT) with PyTorch
===============================================

Here, we provide the steps that are required to integrate QAT from NNCF into the training script written with
PyTorch:

1. Import NNCF API
########################

In this step, you add NNCF-related imports in the beginning of the training script:

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [imports]


2. Create the data transformation function
################################################

In the next step, you need to create a quantization data loader and wrap it by the nncf.Dataset, specifying a transformation
function which prepares input data to fit into model during quantization.

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [nncf_dataset]


3. Apply optimization methods
####################################

nncf.quantize function accepts model and prepared quantization dataset for performing basic quantization. Optionally,
additional parameters like subset_size, preset, ignored_scope can be provided to improve quantization result if applicable.
More details about supported parameters can be found on this `page <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters>`__.

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [quantize]


4. Fine-tune the model
########################

This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the
case of QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5. In principle,
you can skip this step which means that the post-training optimization will be applied to the model.

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [tune_model]



5. Export quantized model
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

7. (Optional) Save checkpoint
####################################

To save model checkpoint use the following API:

.. tab:: PyTorch

   .. doxygensnippet:: docs/optimization_guide/nncf/code/qat_torch.py
      :language: python
      :fragment: [save_checkpoint]


8. (Optional) Restore from checkpoint
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
