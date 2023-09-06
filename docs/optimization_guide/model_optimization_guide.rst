.. {#openvino_docs_model_optimization_guide}

Model Optimization Guide
========================


.. toctree::
   :maxdepth: 1
   :hidden:

   ptq_introduction
   tmo_introduction
   weight_compression


Model optimization is an optional offline step of improving the final model performance and reducing the model size by applying special optimization methods, such as 8-bit quantization, pruning, etc. OpenVINO offers two optimization paths implemented in `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__:

- :doc:`Post-training Quantization <ptq_introduction>` is designed to optimize the inference of deep learning models by applying the post-training 8-bit integer quantization that does not require model retraining or fine-tuning.

- :doc:`Training-time Optimization <tmo_introduction>`, a suite of advanced methods for training-time model optimization within the DL framework, such as PyTorch and TensorFlow 2.x. It supports methods like Quantization-aware Training, Structured and Unstructured Pruning, etc. 

- :doc:`Weight Compression <weight_compression>`, an easy-to-use method for Large Language Models footprint reduction and inference acceleration.

.. note:: OpenVINO also supports optimized models (for example, quantized) from source frameworks such as PyTorch, TensorFlow, and ONNX (in Q/DQ; Quantize/DeQuantize format). No special steps are required in this case and optimized models can be converted to the OpenVINO Intermediate Representation format (IR) right away.

Post-training Quantization is the fastest way to optimize an arbitrary DL model and should be applied first, but it is limited in terms of achievable accuracy-performance trade-off. The recommended approach to obtain OpenVINO quantized model is to convert a model from original framework to ``ov.Model`` and ensure that the model works correctly in OpenVINO, for example, by calculating the model metrics. Then, ``ov.Model`` can be used as input for the ``nncf.quantize()`` method to get the quantized model (see the diagram below).

In case of unsatisfactory accuracy or performance after Post-training Quantization, Training-time Optimization can be used as an option.

.. image:: _static/images/DEVELOPMENT_FLOW_V3_crunch.svg

Once the model is optimized using the aforementioned methods, it can be used for inference using the regular OpenVINO inference workflow. No changes to the inference code are required.

.. image:: _static/images/WHAT_TO_USE.svg

Additional Resources
####################

- :doc:`Post-training Quantization <ptq_introduction>`
- :doc:`Training-time Optimization <tmo_introduction>`
- :doc:`Weight Compression <weight_compression>`
- :doc:`Deployment optimization <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`
- `HuggingFace Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__

