# Optimize Preprocessing Computation{#openvino_docs_MO_DG_Additional_Optimization_Use_Cases}

Model Optimizer performs preprocessing to a model. It is possible to optimize this step and improve first inference time, to do that, follow the tips bellow:

-	**Image mean/scale parameters**<br>
	Make sure to use the input image mean/scale parameters (`--scale` and `–mean_values`) with the Model Optimizer when you need pre-processing. It allows the tool to bake the pre-processing into the IR to get accelerated by the Inference Engine.

-	**RGB vs. BGR inputs**<br>
	If, for example, your network assumes the RGB inputs, the Model Optimizer can swap the channels in the first convolution using the `--reverse_input_channels` command line option, so you do not need to convert your inputs to RGB every time you get the BGR image, for example, from OpenCV*.

-	**Larger batch size**<br>
	Notice that the devices like GPU are doing better with larger batch size. While it is possible to set the batch size in the runtime using the Inference Engine [ShapeInference feature](../../OV_Runtime_UG/ShapeInference.md).

-	**Resulting IR precision**<br>
The resulting IR precision, for instance, `FP16` or `FP32`, directly affects performance. As CPU now supports `FP16` (while internally upscaling to `FP32` anyway) and because this is the best precision for a GPU target, you may want to always convert models to `FP16`. Notice that this is the only precision that Intel&reg; Movidius&trade; Myriad&trade; 2 and Intel&reg; Myriad&trade; X VPUs support.
