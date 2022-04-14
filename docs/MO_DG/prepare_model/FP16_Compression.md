# Compression of a Model to FP16 {#openvino_docs_MO_DG_FP16_Compression}

Model Optimizer can convert all floating-point weights to `FP16` data type. The resulting IR is called
compressed `FP16` model.

To compress the model, use the `--data_type` option:

```
 mo --input_model INPUT_MODEL --data_type FP16
```

> **NOTE**: Using `--data_type FP32` will give no result and will not force `FP32` 
> precision in the model. If the model was `FP16`, it will have `FP16` precision in IR as well.

The resulting model will occupy about twice as less space in the file system, but it may have some accuracy drop.
Still, degradation of accuracy is negligible for the majority of models. 
Refer to the [Working with devices](../../OV_Runtime_UG/supported_plugins/Device_Plugins.md) page for details on how plugins handle compressed `FP16` models.

> **NOTE**: `FP16` compression is sometimes used as initial step for `INT8` quantization.
> Refer to the [Post-training optimization](../../../tools/pot/docs/Introduction.md) guide for more information about that.
