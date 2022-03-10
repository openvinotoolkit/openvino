# Embedding Preprocessing Computation {#openvino_docs_MO_DG_Additional_Optimization_Use_Cases}

Input data for inference can be different from training dataset and requires additional preprocessing before inference.
In order to accelerate the whole pipeline including preprocessing and inference, Model Optimizer provides special parameters such as `--mean_values`,
`--scale_values`, `--reverse_input_channels`, and `--layout`. Based on these parameters, Model Optimizer generates IR with additionally
inserted sub-graph that performs defined preprocessing. This preprocessing block can perform mean-scale normalization of input data,
reverting data along channel dimension, and changing layout of data. For more details about these parameters, refer to paragraphs below.

## When to Specify Layout

Layout defines which dimensions are located in shape and can be specified
for both inputs and outputs. For layout syntax check the [Layout API overview](../../OV_Runtime_UG/layout_overview.md). 
To specify layout, you can use either `--layout` or `--source_layout` followed by the layout value. 
Additionally, if a model has more than one input or needs both input and output layouts specified,
each layout needs to be given a name. Here are examples of the two cases:
* `NCHW` layout specified for a model with a single input:

```
mo --input_model /path/to/model --source_layout nchw
mo --input_model /path/to/model --layout nchw
```

* `NCHW` and `NC` layouts specified for a model with two inputs:

```
mo --input_model /path/to/model --source_layout name1(nchw),name2(nc)
mo --input_model /path/to/model --layout name1(nchw),name2(nc)
```

Some preprocessing may require setting of input layouts, for example: batch setting,
application of mean or scales, and reversing input channels (BGR<->RGB).

## How to Change Layout of a Model 

Changing model layout may be necessary if it differs from the one presented by input data. 
Similarly to specifying layout, to change it you can use one of two commands: either
`--layout` or `--target_layout`. Again, if a model has more than one input or needs both input
and output layouts specified, each layout needs to be given a name. Here are examples of the
two cases:
* `NHWC` layout of a single-input model changed to `NCHW`:

```
mo --input_model /path/to/model --source_layout nhwc --target_layout nchw
mo --input_model /path/to/model --layout "nhwc->nchw"
```

* `NHWC` layout of a multi-input model changed to `NCHW`:

```
mo --input_model /path/to/model --source_layout name1(nhwc),name2(nc) --target_layout name1(nchw)
mo --input_model /path/to/model --layout "name1(nhwc->nchw),name2(nc)"
```

## When to Specify Mean and Scale Values
Usually neural network models are trained with the normalized input data. This means that the input data values are converted to be in a specific range,
for example, `[0, 1]` or `[-1, 1]`. Sometimes the mean values (mean images) are subtracted from the input data values as part of the pre-processing.
There are two cases how the input data pre-processing is implemented.
 * The input pre-processing operations are a part of a topology. In this case, the application that uses the framework to infer the topology does not pre-process the input.
 * The input pre-processing operations are not a part of a topology and the pre-processing is performed within the application which feeds the model with an input data.

In the first case, the Model Optimizer generates the IR with required pre-processing operations and OpenVINO Samples may be used to infer the model.

In the second case, information about mean/scale values should be provided to the Model Optimizer to embed it to the generated IR.
Model Optimizer provides a number of command line parameters to specify them: `--mean_values`, `--scale_values`, `--scale`.

> **NOTE:** If both mean and scale values are specified, the mean is subtracted first and then scale is applied regardless of the order of options
in command line. Input values are *divided* by the scale value(s). If also `--reverse_input_channels` option is used, the reverse_input_channels
will be applied first, then mean and after that scale.

There is no a universal recipe for determining the mean/scale values for a particular model. The steps below could help to determine them:
* Read the model documentation. Usually the documentation describes mean/scale value if the pre-processing is required.
* Open the example script/application executing the model and track how the input data is read and passed to the framework.
* Open the model in a visualization tool and check for layers performing subtraction or multiplication (like `Sub`, `Mul`, `ScaleShift`, `Eltwise` etc)
of the input data. If such layers exist, pre-processing is probably part of the model.

For example, run the Model Optimizer for the PaddlePaddle* UNet model and apply mean-scale normalization to the input data.

```sh
mo --input_model unet.pdmodel --input data --mean_values data[123,117,104] --scale_values data[255,255,255]
```

## When to Reverse Input Channels <a name="when_to_reverse_input_channels"></a>
Input data for your application can be of RGB or BRG color input order. For example, OpenVINO Samples load input images in the BGR channels order.
However, the model may be trained on images loaded with the opposite order (for example, most TensorFlow\* models are trained with images in RGB order).
In this case, inference results using the OpenVINO samples may be incorrect. The solution is to provide `--reverse_input_channels` command line parameter.
Taking this parameter, the Model Optimizer performs first convolution or other channel dependent operation weights modification so these operations output
will be like the image is passed with RGB channels order.

For example, launch the Model Optimizer for the TensorFlow* AlexNet model with reversed input channels order between RGB and BGR.

```sh
mo --input_model alexnet.pb --reverse_input_channels
```
