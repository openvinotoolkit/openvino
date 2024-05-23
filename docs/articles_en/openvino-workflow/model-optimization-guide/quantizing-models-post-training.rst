Quantizing Models Post-training
===============================

.. toctree::
   :maxdepth: 1
   :hidden:

   quantizing-models-post-training/basic-quantization-flow
   quantizing-models-post-training/quantizing-with-accuracy-control


Post-training model optimization is the process of applying special methods that transform a
model into a more hardware-friendly representation, without retraining or fine-tuning it. The
most widely-adopted method is **8-bit post-training quantization** because it is:

* easy-to-use
* does not impact accuracy much
* provides significant performance improvement
* fits most hardware, since 8-bit computation is widely supported

8-bit integer quantization lowers the precision of weights and activations to 8 bits. This
leads to almost 4x reduction in the model footprint and significant improvements in inference
speed, mostly due to reduced throughput. The reduction is performed before the actual inference,
when the model gets transformed into the quantized representation. The process does not require
any training datasets or pipelines in the source DL framework.

.. image:: ../../assets/images/quantization_picture.svg

`Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__
provides a post-training quantization API, available in Python, that aims at reusing the code for
model training or validation that is usually available with the model in the source framework,
such as PyTorch or TensroFlow. The NNCF API is cross-framework and currently supports:
OpenVINO, PyTorch, TensorFlow 2.x, and ONNX. Post-training quantization for models in the
:doc:`OpenVINO IR format <../../documentation/openvino-ir-format>` is the most mature in terms
of supported methods and model coverage.

The NNCF API offers two main options to apply 8-bit post-training quantization:

* :doc:`Basic quantization <quantizing-models-post-training/basic-quantization-flow>` -
  the simplest quantization flow that allows applying 8-bit integer quantization to the model. A
  representative calibration dataset is only needed in this case.
* :doc:`Quantization with accuracy control <quantizing-models-post-training/quantizing-with-accuracy-control>` -
  the most advanced quantization flow that allows applying 8-bit quantization to the model with
  accuracy control. Calibration and validation datasets, and a validation function to calculate
  the accuracy metric are needed in this case.

Additional Resources
####################

* :doc:`Optimizing Models at Training Time <compressing-models-during-training>`
* `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__

