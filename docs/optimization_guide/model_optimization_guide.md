# Model Optimization Guide {#openvino_docs_model_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ptq_introduction
   tmo_introduction


Model optimization is an optional offline step of improving the final model performance and reducing the model size by applying special optimization methods, such as 8-bit quantization, pruning, etc. OpenVINO offers two optimization paths implemented in `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__:

- :doc:`Post-training Quantization <ptq_introduction>` is designed to optimize the inference of deep learning models by applying the post-training 8-bit integer quantization that does not require model retraining or fine-tuning.

- :doc:`Training-time Optimization <tmo_introduction>`, a suite of advanced methods for training-time model optimization within the DL framework, such as PyTorch and TensorFlow 2.x. It supports methods like Quantization-aware Training, Structured and Unstructured Pruning, etc. 

.. note:: OpenVINO also supports optimized models (for example, quantized) from source frameworks such as PyTorch, TensorFlow, and ONNX (in Q/DQ; Quantize/DeQuantize format). No special steps are required in this case and optimized models can be converted to the OpenVINO Intermediate Representation format (IR) right away.

Post-training Quantization is the fastest way to optimize a model and should be applied first, but it is limited in terms of achievable accuracy-performance trade-off. In case of poor accuracy or performance after Post-training Quantization, Training-time Optimization can be used as an option.

Once the model is optimized using the aforementioned methods, it can be used for inference using the regular OpenVINO inference workflow. No changes to the inference code are required.

.. image:: _static/images/DEVELOPMENT_FLOW_V3_crunch.svg

.. image:: _static/images/WHAT_TO_USE.svg

Additional Resources
####################

- :doc:`Post-training Quantization <ptq_introduction>`
- :doc:`Training-time Optimization <tmo_introduction>`
- :doc:`Deployment optimization <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`
- `HuggingFace Optimum Intel <https://huggingface.co/docs/optimum/intel/optimization_ov>`__

@endsphinxdirective
