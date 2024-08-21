Model Optimization - NNCF
=========================================================================================


.. toctree::
   :maxdepth: 1
   :hidden:

   model-optimization-guide/nncf
   model-optimization-guide/quantizing-models-post-training
   model-optimization-guide/compressing-models-during-training
   model-optimization-guide/weight-compression


Model optimization means altering the model itself to improve the inference performance.
It may be done by reducing the model size, applying methods such as 8-bit quantization or
pruning. It is an optional step and should be performed outside of the final software
application.

:doc:`Neural Network Compression Framework (NNCF) <model-optimization-guide/nncf>`
Is the the OpenVINO toolkit's optimisation tool, offering three ways to get more
streamlined models:

- :doc:`Post-training Quantization <model-optimization-guide/quantizing-models-post-training>`
  is designed to optimize inference of deep learning models by applying 8-bit integer
  quantization, which is done post-training and does not require model retraining or
  fine-tuning.

- :doc:`Training-time Optimization <model-optimization-guide/compressing-models-during-training>`
  involves a suite of advanced methods such as Quantization-aware Training, Structured,
  and Unstructured Pruning They are used within the model's deep learning framework, such
  as PyTorch and TensorFlow.

- :doc:`Weight Compression <model-optimization-guide/weight-compression>`
  is an easy-to-use method for Large Language Model footprint reduction and inference
  acceleration.

.. image:: ../assets/images/WHAT_TO_USE.svg

Post-training Quantization is the fastest way to optimize an arbitrary deep learning model
and should be applied first. But it is limited in terms of how much you can increase
performance without significantly impacting the accuracy.

The recommended approach to obtain an OpenVINO quantized model is to convert
`a model from its original framework <https://huggingface.co/models>`__ to ``ov.Model``
and ensure that it works correctly in OpenVINO. You can calculate the model metrics
to do so. Then, ``ov.Model`` can be used as input for the ``nncf.quantize()`` method
to get the quantized model or as input for the ``nncf.compress_weights()`` method to
compress weights, in the case of Large Language Models (see the diagram below).

If Post-training Quantization produces unsatisfactory accuracy or performance results,
Training-time Optimization may prove a better option.

.. image:: ../assets/images/DEVELOPMENT_FLOW_V3_crunch.svg


.. note::

   Once optimized, models may be executed with the typical OpenVINO inference workflow,
   no additional changes to the inference code are required.

   This is true for models optimized using NNCF, as well as those pre-optimized from
   their source frameworks (e.g., quantized), such as PyTorch, TensorFlow, and ONNX
   (in Q/DQ; Quantize/DeQuantize format). The latter may be easily converted to the
   :doc:`OpenVINO Intermediate Representation format (IR) <../../documentation/openvino-ir-format>`
   right away.

Additional Resources
####################

- :doc:`Post-training Quantization <model-optimization-guide/quantizing-models-post-training>`
- :doc:`Training-time Optimization <model-optimization-guide/compressing-models-during-training>`
- :doc:`Weight Compression <model-optimization-guide/weight-compression>`
- :doc:`Deployment optimization <running-inference/optimize-inference>`
- `Hugging Face Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__

