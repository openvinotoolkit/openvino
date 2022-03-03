# Optimize Preprocessing Computation{#openvino_docs_MO_DG_Additional_Optimization_Use_Cases}

Model Optimizer performs preprocessing to a model. It is possible to optimize this step and improve first inference time, to do that, follow the tips bellow.
The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the `RGB<->BGR` conversion specifying the command-line parameter: `--reverse_input_channels`. Otherwise, inference results may be incorrect. For details, refer to [When to Reverse Input Channels](#when_to_reverse_input_channels).

-	**Image mean/scale parameters**<br>
	Make sure to use the input image mean/scale parameters (`--scale` and `–mean_values`) with the Model Optimizer when you need pre-processing. It allows the tool to bake the pre-processing into the IR to get accelerated by the OpenVINO Runtime.

-	**RGB vs. BGR inputs**<br>
	If, for example, your network assumes the RGB inputs, the Model Optimizer can swap the channels in the first convolution using the `--reverse_input_channels` command line option, so you do not need to convert your inputs to RGB every time you get the BGR image, for example, from OpenCV*.

-	**Larger batch size**<br>
	Notice that the devices like GPU are doing better with larger batch size. While it is possible to set the batch size in the runtime using the OpenVINO Runtime API [ShapeInference feature](../../OV_Runtime_UG/ShapeInference.md).

-	**Resulting IR precision**<br>
The resulting IR precision, for instance, `FP16` or `FP32`, directly affects performance. As CPU now supports `FP16` (while internally upscaling to `FP32` anyway) and because this is the best precision for a GPU target, you may want to always convert models to `FP16`. Notice that this is the only precision that Intel&reg; Movidius&trade; Myriad&trade; 2 and Intel&reg; Myriad&trade; X VPUs support.

## When to Specify Mean and Scale Values
Usually neural network models are trained with the normalized input data. This means that the input data values are converted to be in a specific range, for example, `[0, 1]` or `[-1, 1]`. Sometimes the mean values (mean images) are subtracted from the input data values as part of the pre-processing. There are two cases how the input data pre-processing is implemented.
 * The input pre-processing operations are a part of a topology. In this case, the application that uses the framework to infer the topology does not pre-process the input.
 * The input pre-processing operations are not a part of a topology and the pre-processing is performed within the application which feeds the model with an input data.
 
In the first case, the Model Optimizer generates the IR with required pre-processing operations and OpenVINO Samples may be used to infer the model. 
 
In the second case, information about mean/scale values should be provided to the Model Optimizer to embed it to the generated IR. Model Optimizer provides a number of command line parameters to specify them: `--mean`, `--scale`, `--scale_values`, `--mean_values`. 

> **NOTE:** If both mean and scale values are specified, the mean is subtracted first and then scale is applied regardless of the order of options in command line. Input values are *divided* by the scale value(s). If also `--reverse_input_channels` option is used, the reverse_input_channels will be applied first, then mean and after that scale.

There is no a universal recipe for determining the mean/scale values for a particular model. The steps below could help to determine them:
* Read the model documentation. Usually the documentation describes mean/scale value if the pre-processing is required.
* Open the example script/application executing the model and track how the input data is read and passed to the framework.
* Open the model in a visualization tool and check for layers performing subtraction or multiplication (like `Sub`, `Mul`, `ScaleShift`, `Eltwise` etc) of the input data. If such layers exist, pre-processing is probably part of the model.

## When to Reverse Input Channels <a name="when_to_reverse_input_channels"></a>
Input data for your application can be of RGB or BRG color input order. For example, OpenVINO Samples load input images in the BGR channels order. However, the model may be trained on images loaded with the opposite order (for example, most TensorFlow\* models are trained with images in RGB order). In this case, inference results using the OpenVINO samples may be incorrect. The solution is to provide `--reverse_input_channels` command line parameter. Taking this parameter, the Model Optimizer performs first convolution or other channel dependent operation weights modification so these operations output will be like the image is passed with RGB channels order.
