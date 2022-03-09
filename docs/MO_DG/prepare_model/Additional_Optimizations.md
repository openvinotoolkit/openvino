# Optimize Preprocessing Computation {#openvino_docs_MO_DG_Additional_Optimization_Use_Cases}

Model Optimizer performs preprocessing to a model. It is possible to optimize this step and improve first inference time, to do that, follow the tips bellow:

-	**Image mean/scale parameters**<br>
	Make sure to use the input image mean/scale parameters (`--scale` and `–mean_values`) with the Model Optimizer when you need pre-processing. It allows the tool to bake the pre-processing into the IR to get accelerated by the OpenVINO Runtime.

-	**RGB vs. BGR inputs**<br>
	If, for example, your network assumes the RGB inputs, the Model Optimizer can swap the channels in the first convolution using the `--reverse_input_channels` command line option, so you do not need to convert your inputs to RGB every time you get the BGR image, for example, from OpenCV*.

-	**Larger batch size**<br>
	Some devices, like GPU, achieve better results with larger batch sizes. In such cases, it is possible to set the batch size using the OpenVINO Runtime API [ShapeInference feature](../../OV_Runtime_UG/ShapeInference.md).

## When to Specify Layout

Layout defines which dimensions located in shape. Layout syntax is explained in
[Layout API overview](../../OV_Runtime_UG/layout_overview.md). It is possible to specify
layout for inputs and outputs. There are 2 options to specify layout `--layout` and
`--source_layout`. For example to specify `NCHW` layout for model with single input:
```
mo --input_model /path/to/model --source_layout nchw
mo --input_model /path/to/model --layout nchw
```
If model has more than 1 input or if not only input layout but also output layout needs
to be specified it is required to specify names for each layout:
```
mo --input_model /path/to/model --source_layout name1(nchw),name2(nc)
mo --input_model /path/to/model --layout name1(nchw),name2(nc)
```
Some preprocessing may require setting of input layouts, for example: batch setting,
application of mean or scales, and reversing input channels (BGR<->RGB).

## How to Change Layout of a Model 

It is possible to change layout of the model. It may be needed if input data has different
layout then model was trained on. There are 2 options that can be used to change layout of
model inputs or outputs --layout and --target_layout. For example to change layout of the
model with one input from NHWC to NCHW:
```
mo --input_model /path/to/model --source_layout nhwc --target_layout nchw
mo --input_model /path/to/model --layout "nhwc->nchw"
```
Similarly, if model has multiple inputs or if it is required to change layout of output,
names must be specified:
```
mo --input_model /path/to/model --source_layout name1(nhwc),name2(nc) --target_layout name1(nchw)
mo --input_model /path/to/model --layout "name1(nhwc->nchw),name2(nc)"
```

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
Input data for your application can be of either RGB or BRG color input order. For example, OpenVINO Samples load input images in the BGR channel order, but many models are trained in the RBG one, like most TensorFlow cases. Such discrepancy may result in incorrect inference results. The solution is to use the `--reverse_input_channels` command line parameter, which tells Model Optimizer to perform weights modification with the first convolution or other channel-dependent operation. Therefore, it will look like the image is passed with the RGB channel order.

## Compression of model to FP16

Model Optimizer can compress models to `FP16` data type. This makes them occupy less space 
in the file system and, most importantly, increase performance when particular hardware is used. 
The process assumes changing data type on all constants inside the model
to the `FP16` precision and inserting `Convert` nodes to the initial data type, so that the data
flow inside the model is preserved. To compress the model to `FP16` use the `--data_type` option like this:

```
mo --input_model /path/to/model --data_type FP16
```

> **NOTE**: Using `--data_type FP32` will not do anything and will not force `FP32` 
> precision in the model. If the model was `FP16` originally in the framework,
> Model Optimizer will not convert such weights to `FP32` even if `--data_type FP32`
> option is used .

Some plugins, for example GPU, will show greater performance while slightly sacrificing
accuracy.

> **NOTE**: Intel&reg; Movidius&trade; Myriad&trade; 2 and Intel&reg; Myriad&trade; X VPUs
> require models in `FP16` precision.

