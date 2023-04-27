# Precision Control {#openvino_docs_OV_UG_Precision_Control}

@sphinxdirective

Regardless of IR precision plugins will execute in high performance mode by default. For GPU this means fp16 inference and for CPU this means bf16 inference (if available of course). Prior to this you had to convert IR to fp16 to get GPU to execute in fp16 (we had issue with stable diffusion repo because of this for example) and for CPU it was high precision by default. Now we have aligned plugins and disconnected this selection from IR precision. If high performance is bringing accuracy issue (spotted only few times in history) then you can use inference_precision hint and set it to accuracy.
Separately, you can control IR precision. By default, we wanted to set it to fp16 to reduce model size 2x for floating point models and failed to do it due to CPU plugin specifics in case of large models that we cannot fix until release. Basically, IR precision is becoming a way to compress your model by reducing weights precision and does not influence how plugins execute model.

This change removes a lot of confusion where you could not execute model with high performance by default in GPU for instance and behavior between plugins was different. I have talked about this during our call.


.. note::

   All devices (except GNA) only support floating-point data types (``f32``, ``f16``, ``bf16``) as value for inference_precision property, as quantization can’t be done in runtime. GNA plugin has a capability to do model quantization on ``core.compile_model()`` call, thus it supports integer data types in addition to ``f32``.

Execution Mode
##############

So ov::hint::execution_mode is a high level hint to control if user want to keep accuracy as good as possible (ACCURACY mode) or a plugin may do some optimizations which may lower the accuracy for the sake of performance (PERFORMANCE mode)

Inference Precision
###################

ov::hint::inference precision is a lower level property which allows to specify exact precision which user wants, but it’s less portable. E.g. CPU supports f32 inference precision and bf16 on some platforms, GPU supports fp32 and fp16 while GNA supports i8 and i16, so if user want to have an app that utilize multiple devices, then he needs to handle all these combinations manually or let OV do it automatically by using higher level execution_mode property. Another thing is that inference_precision is also a hint, so the specified value is not guaranteed to be used by runtime (mostly in cases when current device doesn’t have required HW capabilities). 

Additional Resources
####################

* :doc:`Supported Devices <openvino_docs_OV_UG_Working_with_devices>`

@endsphinxdirective

