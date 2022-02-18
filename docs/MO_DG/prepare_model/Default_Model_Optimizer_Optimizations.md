# Default Model Optimizer Optimizations {#openvino_docs_MO_DG_Default_Model_Optimizer_Optimizations}

Model Optimizer not only converts a model to IR format, but also performs a number of optimizations. For example, certain primitives like linear operations (BatchNorm and ScaleShift), are automatically fused into convolutions. Generally, these layers should not be manifested in the resulting IR:

![](../img/optimizations/resnet_269.png)

The picture above shows Caffe\* Resnet269\* topology. The left model is the original model, and the one on the right (after conversion) is the resulting model that the Model Optimizer produces, with BatchNorm and ScaleShift layers  fused into the convolution weights rather than constituting separate layers.

If you still see these operations, inspect the Model Optimizer output carefully while searching for warnings, such as on the tool being unable to fuse. For example, non-linear operations (like activations) in between convolutions and linear operations might prevent the fusing. If performance is of concern, try to change (and potentially re-train) the topology. Refer to the [Model Optimizer Guide](Model_Optimization_Techniques.md) for more optimizations.

Notice that the activation (`_relu`) is not touched by the Model Optimizer, and while it can be merged into convolution as well, this is rather a device-specific optimization, covered by Inference Engine during the model loading time. You are encouraged to inspect performance counters from plugins that should indicate that these particular layers are not executed (“Optimized out”). For more information, refer to <a href="#performance-counters">Internal Inference Performance Counters</a>.
