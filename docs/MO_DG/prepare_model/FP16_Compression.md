# Compression of a Model to FP16 {#openvino_docs_MO_DG_FP16_Compression}

Model Optimizer can convert all floating point weights to `FP16` data type. The resulting IR is called
compressed `FP16` model.

To compress the model use the `--data_type` option like this:

```
mo --input_model /path/to/model --data_type FP16
```

> **NOTE**: Using `--data_type FP32` will give no result and will not force `FP32` 
> precision in the model. If the model was `FP16` it will have `FP16` precision in IR as well.

The resulting model will occupy about twice as less space in the file system, but it will have slightly worse accuracy
on hardware that support `FP16` inference, although for the majority of models accuracy degradation is negligible. 
For details about which plugins can utilize inference in `FP16` please refer to each plugin 
documentation: [Working with devices](../../OV_Runtime_UG/supported_plugins/Device_Plugins.md).

> **NOTE**: `FP16` compression is sometimes used as initial step for `INT8` quantization, please refer to
> [Post-Training Optimization tool](../../../tools/pot/README.md) for more information about that.
