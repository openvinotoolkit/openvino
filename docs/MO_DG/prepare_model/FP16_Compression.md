# Compression Of Model To FP16 {#openvino_docs_MO_DG_FP16_Compression}

Model Optimizer can compress models to `FP16` data type. This makes them occupy less space 
in the file system and, most importantly, increase performance when particular hardware is used. 
The process assumes changing data type on all constants inside the model
to the `FP16` precision and inserting `Convert` nodes to the initial data type, so that the data
flow inside the model is preserved. To compress the model to `FP16` use the `--data_type` option like this:

```
mo --input_model /path/to/model --data_type FP16
```

> **NOTE**: Using `--data_type FP32` will give no result and will not force `FP32` 
> precision in the model. If the model was `FP16` it will have `FP16` precision in IR as well.

Some plugins, for example GPU, will show greater performance while slightly sacrificing
accuracy.

> **NOTE**: Intel&reg; Movidius&trade; Myriad&trade; 2 and Intel&reg; Myriad&trade; X VPUs
> require models in `FP16` precision.

