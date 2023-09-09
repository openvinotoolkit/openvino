# Quantizing Models Post-training {#ptq_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   basic_quantization_flow
   quantization_w_accuracy_control
   

Post-training model optimization is the process of applying special methods that transform the model into a more hardware-friendly representation without retraining or fine-tuning. The most popular and widely-spread method here is 8-bit post-training quantization because it is:

* It is easy-to-use.
* It does not hurt accuracy a lot.
* It provides significant performance improvement.
* It suites many hardware available in stock since most of them support 8-bit computation natively.

8-bit integer quantization lowers the precision of weights and activations to 8 bits, which leads to almost 4x reduction in the model footprint and significant improvements in inference speed, mostly due to lower throughput required for the inference. This lowering step is done offline, before the actual inference, so that the model gets transformed into the quantized representation. The process does not require a training dataset or a training pipeline in the source DL framework.

.. image:: _static/images/quantization_picture.svg

`Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides a post-training quantization API available in Python that is aimed at reusing the code for model training or validation that is usually available with the model in the source framework, for example, PyTorch or TensroFlow. The NNCF API is cross-framework and currently supports models in the following frameworks: OpenVINO, PyTorch, TensorFlow 2.x, and ONNX. Currently, post-training quantization for models in OpenVINO Intermediate Representation is the most mature in terms of supported methods and models coverage. 

NNCF API has two main capabilities to apply 8-bit post-training quantization:

* :doc:`Basic quantization <basic_quantization_flow>` - the simplest quantization flow that allows applying 8-bit integer quantization to the model. A representative calibration dataset is only needed in this case.
* :doc:`Quantization with accuracy control <quantization_w_accuracy_control>` - the most advanced quantization flow that allows applying 8-bit quantization to the model with accuracy control. Calibration and validation datasets, and a validation function to calculate the accuracy metric are needed in this case.

Additional Resources
####################

* :doc:`Optimizing Models at Training Time <tmo_introduction>`
* `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__
* `Tutorial: Migrate quantization from POT API to NNCF API <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-yolov5-quantization-migration>`__

@endsphinxdirective
