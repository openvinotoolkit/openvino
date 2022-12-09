# Post-training Quantization w/ NNCF (new) {#nncf_ptq_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   basic_qauntization_flow
   quantization_w_accuracy_control

@endsphinxdirective

Neural Network Compression Framework (NNCF) provides a new post-training quantization API available in Python that is aimed at reusing the code for model training or validation that is usually available with the model in the source framework, for example, PyTorch* or TensroFlow*. The API is cross-framework and currently supports models representing in the following frameworks: PyTorch, TensorFlow 2.x, ONNX, and OpenVINO. 
This API has two main capabilities to apply 8-bit post-training quantization:
* [Basic quantization](@ref basic_qauntization_flow) - the simplest quantization flow that allows to apply 8-bit integer quantization to the model.
* [Quantization with accuracy control](@ref quantization_w_accuracy_control) - the most advanced quantization flow that allows to apply 8-bit quantization to the model with accuracy control.

## See also

* [NNCF GitHub](https://github.com/openvinotoolkit/nncf)
* [Optimizing Models at Training Time](@ref tmo_introduction)