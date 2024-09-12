Post-training Quantization
===============================

.. toctree::
   :maxdepth: 1
   :hidden:

   quantizing-models-post-training/basic-quantization-flow
   quantizing-models-post-training/quantizing-with-accuracy-control


Post-training quantization is a method of reducing the size of a model, to make it lighter,
faster, and less resource hungry. Importantly, this process does not require retraining,
fine-tuning, or using training datasets and pipelines in the source framework. With NNCF, you
can perform `8-bit quantization <#why-8-bit-post-training-quantization>`__, using mainly the two
flows:

| :doc:`Basic quantization (simple) <quantizing-models-post-training/basic-quantization-flow>`:
|     Requires only a representative calibration dataset.

| :doc:`Accuracy-aware Quantization (advanced) <quantizing-models-post-training/quantizing-with-accuracy-control>`:
|     Ensures the accuracy of the resulting model does not drop below a certain value.
      To do so, it requires both a calibration and a validation datasets, as well as a
      validation function to calculate the accuracy metric.

.. note

   NNCF offers a Python API, for compressing PyTorch, TensorFlow 2.x, ONNX, and OpenVINO IR
   model formats. OpenVINO IR offers the most comprehensive support.


Why 8-bit post-training quantization
####################################

The 8-bit quantization is just one of the available compression methods but one often
selected for:

* significant performance results,
* little impact on accuracy,
* ease of use,
* wide hardware compatibility.

It lowers model weight and activation precisions to 8 bits (INT8), which for an FP64 model
is just a quarter of the original footprint, leading to a significant improvement in inference
speed.


.. image:: ../../assets/images/quantization_picture.svg



Additional Resources
####################

* :doc:`Optimizing Models at Training Time <compressing-models-during-training>`
* :doc:`Model Optimization - NNCF <../model-optimization>`
* `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__
